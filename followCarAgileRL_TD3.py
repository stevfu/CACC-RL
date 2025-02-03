import gymnasium as gym
from agilerl.utils.utils import make_vect_envs
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.algorithms.td3 import TD3
from agilerl.utils.utils import create_population, make_vect_envs
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_off_policy import train_off_policy
import numpy as np
import torch

import gym_followCar

'''
Description: 

Use of AgileRL library to train agents based on TD3 algorithm. 

'''


# Create Environment and Experience Replay Buffer

def main():
    

    # Hyperparameters

    INIT_HP = {
        "MAX_ACTION" : float(env.single_action_space.high[0]),
        "MIN_ACTION" : float(env.single_action_space.low[0]),

        "ALGO": "TD3",
        "POP_SIZE": 4,  # Population size
        "BATCH_SIZE": 256,  # Batch size
        "LR_ACTOR": 0.0001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "O_U_NOISE": False,  # Ornstein-Uhlenbeck action noise
        "EXPL_NOISE": 0.2,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 100_000,  # Max memory buffer size
        "POLICY_FREQ": 3,  # Policy network update frequency
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 0.005,  # For soft update of target parameters
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,  # Use with RGB states
        "EPISODES": 10000,  # Number of episodes to train for
        "EVO_EPOCHS": 20,  # Evolution frequency, i.e. evolve after every 20 episodes
        "TARGET_SCORE": 500.0,  # Target score that will beat the environment
        "EVO_LOOP": 3,  # Number of evaluation episodes
        "MAX_STEPS": 2000,  # Maximum number of steps an agent takes in an environment
        "LEARNING_DELAY": 10000,  # Steps before starting learning
        "EVO_STEPS": 10000,  # Evolution frequency
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
        # Learning HPs to choose from
        "RL_HP_SELECTION": ["lr", "batch_size", "learn_step"],
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

    ## Check to see if environment is using cont. or discrete observation and action spaces 
    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      
    except Exception:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False
                                                       
    try:
        action_dim = env.single_action_space.n              # Discrete action space
    except:
        action_dim = env.single_action_space.shape[0]       # Continuous action space

    channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

    if channels_last:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])


    # Create a Population of Agents
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    net_config = {"arch": "mlp", "hidden_size": [64,64]}

    pop = create_population(
        algo="TD3",  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=net_config,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,
        device=device,
    )

    # Tournament Initialization

    tournament = TournamentSelection(
    INIT_HP["TOURN_SIZE"],
    INIT_HP["ELITISM"],
    INIT_HP["POP_SIZE"],
    INIT_HP["EVAL_LOOP"],
    )   

    # Mutation Initialization 

    mutations = Mutations(
    algo=INIT_HP["ALGO"],
    no_mutation=MUT_P["NO_MUT"],
    architecture=MUT_P["ARCH_MUT"],
    new_layer_prob=MUT_P["NEW_LAYER"],
    parameters=MUT_P["PARAMS_MUT"],
    activation=MUT_P["ACT_MUT"],
    rl_hp=MUT_P["RL_HP_MUT"],
    rl_hp_selection=MUT_P["RL_HP_SELECTION"],
    min_lr=MUT_P["MIN_LR"],
    max_lr=MUT_P["MAX_LR"],
    min_batch_size=MUT_P["MIN_BATCH_SIZE"],
    max_batch_size=MUT_P["MAX_BATCH_SIZE"],
    min_learn_step=MUT_P["MIN_LEARN_STEP"],
    max_learn_step=MUT_P["MAX_LEARN_STEP"],
    mutation_sd=MUT_P["MUT_SD"],
    arch=net_config["arch"],
    rand_seed=MUT_P["RAND_SEED"],
    device=device,
    )

    # Set-up replay buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=100000, 
        field_names=field_names,
        device=device 
        )

    # Built-in off policy function to train agent 
    
    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name='followCar-v0',
        algo="TD3",
        pop=pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUT_P,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=False,  # Boolean flag to record run with Weights & Biases
        save_elite=True,  # Boolean flag to save the elite agent in the population
        elite_path="AgileRL_TD3_trained_agent.pt",
    )   

    torch.save(trained_pop[0], "AgileRL_TD3_trained_agent_MANUAL.pt")  # Save the best agent in the population
    print("✅ Best agent saved successfully!")
    
    
    # Previous custom training loop 
    '''
    agent = TD3(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot, max_action=max_action)   # Create TD3 agent

    state = env.reset()[0]  # Reset environment at start of episode

    while True:
        if channels_last:
            state = np.moveaxis(state, [-1], [-3])
        action = agent.get_action(state, training=True)    # Get next action from agent
        next_state, reward, done, _, _ = env.step(action)   # Act in environment


        # Save experience to replay buffer
        if channels_last:
            memory.save_to_memory_vect_envs(state, action, reward, np.moveaxis(next_state, [-1], [-3]), done)
        else:
            memory.save_to_memory_vect_envs(state, action, reward, next_state, done)

        # Learn according to learning frequency
        if len(memory) >= agent.batch_size:
            experiences = memory.sample(agent.batch_size) # Sample replay buffer
            agent.learn(experiences)    # Learn according to agent's RL algorithm
    '''
        

if __name__ == "__main__":
    num_envs = 8
    env = make_vect_envs('followCar-v0',num_envs = num_envs)
    main()