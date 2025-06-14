import os
import time 
import math 
from typing import List

import numpy as np
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

def has_nan_or_inf(d):
    if isinstance(d, dict):
        return any(has_nan_or_inf(v) for v in d.values())
    arr = np.array(d)
    return np.isnan(arr).any() or np.isinf(arr).any()

from pettingzoo.utils import ParallelEnv

class EarlyTerminationWrapper(ParallelEnv):
    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = env.possible_agents
        self.agents = env.agents

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.agents = self.env.agents
        return obs, info

    def step(self, actions):
        obs, reward, termination, truncation, info = self.env.step(actions)
        # If any agent is done, set all to done
        if any(termination.values()) or any(truncation.values()):
            for agent in self.env.agents:
                termination[agent] = True
                truncation[agent] = truncation.get(agent, False)
        return obs, reward, termination, truncation, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # Define the network configuration
    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [128, 128],  # Actor hidden size
        },
        "head_config": {
            "hidden_size": [128, 128],  # Critic hidden size
        },
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 4,
        "ALGO": "MATD3",  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 128,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.2,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.2,  # Rate of mean reversion in OU noise
        "DT": 0.05,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 100,  # Learning frequency
        "TAU": 0.005,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
    }

    num_envs = 32
    num_followers = 3
    # Define the simple speaker listener environment as a parallel environment
    env = AsyncPettingZooVecEnv(
    [
        lambda: ParallelCarEnv(n_followers=num_followers, render_mode="None")
        for _ in range(num_envs)
    ]
)
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
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_steps = 8000000 # Max steps 
    learning_delay = 2000  # Steps before starting learning
    evo_steps = 2000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 3 # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent

    total_steps = 0

    # --- Load checkpoint if resuming ---
    resume_path = "./trained_agent/MATD3/multiCarAgent_10.pt"
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
                obs, info = env.reset()  # Reset environment at start of episode

                if has_nan_or_inf(obs):
                    print("NaN or Inf detected in obs after reset!")

                scores = np.zeros(num_envs)
                completed_episode_scores = []
                steps = 0
                if INIT_HP["CHANNELS_LAST"]:
                    obs = {
                        agent_id: obs_channels_to_first(s) for agent_id, s in obs.items()
                    }

                
                for idx_step in range(evo_steps // num_envs):

                    # Get next action from agent
                    cont_actions, discrete_action = agent.get_action(
                        obs=obs, training=True, infos=info
                    )
                    if agent.discrete_actions:
                        action = discrete_action
                    else:
                        action = cont_actions

                    # Act in environment
                    next_obs, reward, termination, truncation, info = env.step(action)
                    #time.sleep(0.1)  # Sleep to allow environment to process

                    scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                    total_steps += num_envs
                    steps += num_envs

                    # Image processing if necessary for the environment
                    if INIT_HP["CHANNELS_LAST"]:
                        next_obs = {
                            agent_id: obs_channels_to_first(ns)
                            for agent_id, ns in next_obs.items()
                        }

                    # Save experiences to replay buffer
                    memory.save_to_memory(
                        obs,
                        cont_actions,
                        reward,
                        next_obs,
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
                    # Handle num_envs > learn step; learn multiple times per step in env
                    elif (
                        len(memory) >= agent.batch_size and memory.counter > learning_delay
                    ):
                        for _ in range(num_envs // agent.learn_step):
                            # Sample replay buffer
                            experiences = memory.sample(agent.batch_size)
                            # Learn according to agent's RL algorithm
                            agent.learn(experiences)

                    obs = next_obs

                    # Calculate scores and reset noise for finished episodes
                    reset_noise_indices = []
                    term_array = np.array(list(termination.values())).transpose()
                    trunc_array = np.array(list(truncation.values())).transpose()
                    for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                        if np.any(d) or np.any(t):
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0
                            reset_noise_indices.append(idx)
                    agent.reset_action_noise(reset_noise_indices)

                    # If any agent is done, break and reset the environment
                    if any(np.any(v) for v in termination.values()) or any(np.any(v) for v in truncation.values()):
                        break
                
                obs, info = env.reset()  # Reset environment after episode ends

                agent.steps[-1] += steps
                pop_episode_scores.append(completed_episode_scores)

        
            # Evaluate population
            fitnesses = []
            for agent in pop:
                base_env = ParallelCarEnv(n_followers=num_followers,render_mode=None)
                test_env = EarlyTerminationWrapper(base_env)
                agent.agent_ids = test_env.agents
                obs, info = test_env.reset()
                fitness = agent.test(
                    test_env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=eval_steps,
                    loop=eval_loop,
                )
                fitnesses.append(fitness)

            mean_scores = [
                (
                    np.mean(episode_scores)
                    if len(episode_scores) > 0
                    else "0 completed episodes"
                )
                for episode_scores in pop_episode_scores
            ]

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

            # --- Progress bar update ---
            current_min_steps = min(agent.steps[-1] for agent in pop)
            pbar.update(current_min_steps - last_min_steps)
            last_min_steps = current_min_steps

        # Save the trained algorithm
        path = "./trained_agent/MATD3"
        filename = "multiCarAgent_11.pt"
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