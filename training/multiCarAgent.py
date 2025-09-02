import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import trange

from agilerl.algorithms import MATD3
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import create_population, observation_space_channels_to_first
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

from gym_followCar.multiCarEnv import ParallelCarEnv

if __name__ == "__main__":

    ########## USER PARAMETERS ##########
    num_envs = 32

    max_steps = 1500000  # More steps needed for multi-agent convergence
    learning_delay = 500   # Reduced delay for faster learning start
    evo_steps = 300   # Shorter episodes for faster evolution cycles
    eval_steps = 200  # Limit evaluation episode length
    eval_loop = 1  # Single evaluation episode for speed
    resume = False
    resume_path = "./trained_agent/MATD3/20250630_multi.pt"  # Path to resume training from

    path = "./trained_agent/MATD3"
    filename = "20250819_multi.pt"

    ####################################


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the network configuration
    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [32, 32],  # Smaller networks for faster training
        },
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 4,  # Moderate increase for better exploration
        "ALGO": "MATD3",  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 128,  # Smaller batch for more conservative updates
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.05,  # Lower noise once good policies are found
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.5,  # Faster noise decay for quicker convergence
        "DT": 0.05,  # Timestep for OU noise
        "LR_ACTOR": 0.0003,  # Much lower learning rate to prevent catastrophic forgetting
        "LR_CRITIC": 0.0006,  # Lower critic learning rate for stability
        "GAMMA": 0.995,  # Higher discount for long-term behavior in CACC
        "MEMORY_SIZE": 200000,  # Larger buffer for more diverse experiences
        "LEARN_STEP": 4,  # Less frequent learning to prevent overfitting
        "TAU": 0.0005,  # Even slower soft update for maximum stability
        "POLICY_FREQ": 2,  # Update policy less frequently than critic
    }

    env = ParallelCarEnv(render_mode=None)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
    env.reset()
    agent_ids = env.possible_agents

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    if INIT_HP["CHANNELS_LAST"]:
        observation_spaces = [
            observation_space_channels_to_first(obs) for obs in observation_spaces
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop: List[MATD3] = create_population(
        INIT_HP["ALGO"],
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        hp_config=hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer
    field_names = ["obs", "action", "reward", "next_obs", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.3,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.3,  # Probability of RL hyperparameter mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,
        device=device,
    )

    elite = pop[0]  # Assign a placeholder "elite" agent
    total_steps = 0

    # Load checkpoint if resuming training
    if resume: 
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}")
            elite.load_checkpoint(resume_path)
            for agent in pop:
                agent.load_checkpoint(resume_path)
            

    # TRAINING LOOP
    try:
        print("Training...")
        pbar = trange(max_steps, unit="step")
        last_min_steps = 0

        while np.less([agent.steps[-1] for agent in pop], max_steps).all():
            pop_episode_scores = []

            for agent in pop:  # Loop through population
                state, info = env.reset()  # Reset environment at start of episode
                scores = np.zeros(num_envs)
                completed_episode_scores = []
                steps = 0

                if INIT_HP["CHANNELS_LAST"]:
                    state = {
                        agent_id: obs_channels_to_first(s) for agent_id, s in state.items()
                    }

                for idx_step in range(evo_steps // num_envs):
                    # Get next action from agent
                    cont_actions, discrete_action = agent.get_action(
                        state, training=True, infos=info
                    )

                    if agent.discrete_actions:
                        action = discrete_action
                    else:
                        action = cont_actions

                    # Act in environment
                    next_state, reward, termination, truncation, info = env.step(action)
                    
                    reward_array = np.array(list(reward.values())).transpose()
                    scores += np.sum(reward_array, axis=-1)
                    total_steps += num_envs
                    steps += num_envs

                    if INIT_HP["CHANNELS_LAST"]:
                        next_state = {
                            agent_id: obs_channels_to_first(ns)
                            for agent_id, ns in next_state.items()
                        }

                    # Save experiences to replay buffer
                    memory.save_to_memory(
                        state,
                        cont_actions,
                        reward,
                        next_state,
                        termination,
                        is_vectorised=True,
                    )
                    
                    # Learn according to learning frequency
                    # Handle learn steps > num_envs
                    if agent.learn_step > num_envs:
                        learn_step = agent.learn_step // num_envs
                        if (
                            idx_step % learn_step == 0
                            and len(memory) >= agent.batch_size
                            and memory.counter > learning_delay
                        ):
                            # Sample replay buffer
                            experiences = memory.sample(agent.batch_size)
                            # Learn according to agent's RL algorithm
                            agent.learn(experiences)
                    elif (
                        len(memory) >= agent.batch_size and memory.counter > learning_delay
                    ):
                        for _ in range(num_envs // agent.learn_step):
                            # Sample replay buffer
                            experiences = memory.sample(agent.batch_size)
                            # Learn according to agent's RL algorithm
                            agent.learn(experiences)

                    state = next_state

                    # Calculate scores and reset noise for finished episodes
                    reset_noise_indices = []
                    term_array = np.array(list(termination.values())).transpose()
                    trunc_array = np.array(list(truncation.values())).transpose()
                    for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                        if np.any(d) or np.any(t):
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])  # Append to the current agent
                            scores[idx] = 0
                            reset_noise_indices.append(idx)
                    agent.reset_action_noise(reset_noise_indices)
                
                agent.steps[-1] += steps
                
                # Add any remaining incomplete episodes to the scores
                for idx in range(num_envs):
                    if scores[idx] != 0:  # If there's an incomplete episode
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])  # Append to the current agent
                
                pop_episode_scores.append(completed_episode_scores)
                
        
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
                (
                    np.mean(episode_scores)
                    if len(episode_scores) > 0
                    else "0 completed episodes"
                )
                for episode_scores in pop_episode_scores
            ]
            
            # Dialog output
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
                if len(agent.scores) > 100:
                    agent.scores = agent.scores[-100:]
                if len(agent.fitness) > 50:
                    agent.fitness = agent.fitness[-50:]

            # Progress bar update
            current_min_steps = min(agent.steps[-1] for agent in pop)
            pbar.update(current_min_steps - last_min_steps)
            last_min_steps = current_min_steps


        # Plot moving average reward (window=10) for each agent
        plt.figure(figsize=(10, 6))
        window = 10
        rewards = np.array(elite.scores)
        print(f"Elite scores length: {len(rewards)}")
        print(f"Elite scores: {rewards}")
        if len(rewards) >= window:
            avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(avg_rewards, label='Elite Agent (moving avg)')
        elif len(rewards) > 0:
            plt.plot(rewards, label='Elite Agent (raw)')
        else:
            print("No scores to plot!")
        plt.xlabel('Episode')
        plt.ylabel(f'Average Reward (last {window} episodes)')
        plt.title('Reward Growth During Training (Elite Agent)')
        plt.legend()
        plt.tight_layout()
        # Save plot as .png named after the .pt file
        plot_filename = os.path.splitext(filename)[0] + '.png'
        plt.savefig(os.path.join(path, plot_filename))
        plt.close()

        # Save the trained algorithm
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        elite.save_checkpoint(save_path)

    except KeyboardInterrupt: 
        print("Training interrupted by user.")
        path = "./trained_agent/MATD3"
        filename = "multiCarAgent_interrupted.pt"
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        elite.save_checkpoint(save_path)
    
    finally:
        pbar.close()
        env.close()