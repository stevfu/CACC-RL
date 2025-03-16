import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    create_population,
    make_vect_envs,
    observation_space_channels_to_first,
)

import gym_followCar

if __name__ == "__main__":  

    all_mean_scores = []
    rolling_avg = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [32, 32],  # Actor hidden size
        }
    }

    INIT_HP = {
        "ALGO": "TD3",
        "POP_SIZE": 4,  # Population size
        "BATCH_SIZE": 32,  # Batch size
        "LR_ACTOR": 0.0001,  # Actor learning rate
        "LR_CRITIC": 0.0005,  # Critic learning rate
        "O_U_NOISE": True,  # Ornstein-Uhlenbeck action noise
        "EXPL_NOISE": 0.2,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.05,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 10_000,  # Max memory buffer size
        "POLICY_FREQ": 1,  # Policy network update frequency
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 0.0025,  # For soft update of target parameters

        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,  # Use with RGB states
        
        "EPISODES": 5,  # Number of episodes to train for
        "EVO_EPOCHS": 1,  # Evolution frequency, i.e. evolve after every 20 episodes
        "TARGET_SCORE": 100.0,  # Target score that will beat the environment
        "EVO_LOOP": 3,  # Number of evaluation episodes
        "MAX_STEPS": 500,  # Maximum number of steps an agent takes in an environment
        "LEARNING_DELAY": 0,  # Steps before starting learning
        "EVO_STEPS":100,  # Evolution frequencys
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
    }

    # Mutation parameters
    MUT_P = {
        # Mutation probabilities
        "NO_MUT": 0.4,  # No mutation
        "ARCH_MUT": 0.2,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0.2,  # Network parameters mutation
        "ACT_MUT": 0.2,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": 42,  # Random seed
        # Define max and min limits for mutating RL hyperparams
        "MIN_LR": 0.0001,
        "MAX_LR": 0.01,
        "MIN_BATCH_SIZE": 8,
        "MAX_BATCH_SIZE": 1024,
        "MIN_LEARN_STEP": 1,
        "MAX_LEARN_STEP": 16,
    }

    num_envs = 2
    env = make_vect_envs("followCar-v1", num_envs=num_envs)  # Create environment
    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if INIT_HP["CHANNELS_LAST"]:
        observation_space = observation_space_channels_to_first(observation_space)

    pop = create_population(
        algo="TD3",  # Algorithm
        observation_space=observation_space,  # Observation space
        action_space=action_space,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized envs
        device=device,
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        no_mutation=MUT_P["NO_MUT"],
        architecture=MUT_P["ARCH_MUT"],
        new_layer_prob=MUT_P["NEW_LAYER"],
        parameters=MUT_P["PARAMS_MUT"],
        activation=MUT_P["ACT_MUT"],
        rl_hp=MUT_P["RL_HP_MUT"],
        mutation_sd=MUT_P["MUT_SD"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )

    max_steps = INIT_HP["MAX_STEPS"] # Max steps
    learning_delay = INIT_HP["LEARNING_DELAY"]

    # Exploration params
    eps_start = 1.0  # Max exploration
    eps_end = 0.1  # Min exploration
    eps_decay = 0.995  # Decay per episode
    epsilon = eps_start

    evo_steps = INIT_HP["EVO_STEPS"]  # Evolution frequency
    eval_steps = INIT_HP["EVAL_STEPS"] # Evaluation steps per episode - go until done
    eval_loop = INIT_HP["EVAL_LOOP"]  # Number of evaluation episodes

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            epsilon = eps_start

            for idx_step in range(evo_steps // num_envs):
                if INIT_HP["CHANNELS_LAST"]:
                    state = obs_channels_to_first(state)

                action = agent.get_action(state, epsilon)  # Get next action from agent
                epsilon = max(
                    eps_end, epsilon * eps_decay
                )  # Decay epsilon for exploration

                # Act in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                scores += np.array(reward)
                steps += num_envs
                total_steps += num_envs

                # Collect scores for completed episodes
                for idx, (d, t) in enumerate(zip(terminated, truncated)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0

                # Save experience to replay buffer
                if INIT_HP["CHANNELS_LAST"]:
                    memory.save_to_memory(
                        state,
                        action,
                        reward,
                        obs_channels_to_first(next_state),
                        terminated,
                        is_vectorised=True,
                    )
                else:
                    memory.save_to_memory(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated,
                        is_vectorised=True,
                    )

                # Learn according to learning frequency
                if memory.counter > learning_delay and len(memory) >= agent.batch_size:
                    for _ in range(num_envs // agent.learn_step):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                state = next_state

            pbar.update(evo_steps // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Reset epsilon start to latest decayed value for next round of population training
        eps_start = epsilon

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            np.mean(episode_scores) if len(episode_scores) > 0 else 0.0
            for episode_scores in pop_episode_scores
        ]

        all_mean_scores.append(np.mean(mean_scores) if mean_scores else 0)

        if len(all_mean_scores) >= 10:
            rolling_avg.append(np.mean(all_mean_scores[-10:]))  # Moving average of last 10 episodes
        else:
            rolling_avg.append(np.mean(all_mean_scores))  # If less than 10, take current avg
        

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    best_agent_path = "trained_agent/agileRL_TD3_followCar_v2.pt"
    agent.save_checkpoint(best_agent_path)

    # Plot rolling avg. of reward over time 
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avg, label="Rolling Avg (Last 10 Episodes)", color="blue")
    plt.xlabel("Training Iterations")
    plt.ylabel("Average Score")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.savefig("trained_agent/avgScore_2")
    plt.show()

    pbar.close()
    env.close()