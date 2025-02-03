import torch
import gymnasium as gym
import agilerl
from agilerl.algorithms.td3 import TD3
from agilerl.utils.utils import make_vect_envs

import stable_baselines3 as sb3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import followCarEnvironment  
import os

env = gym.make("followCar-v0",render_mode = "human")

agentType = "sb3"

if agentType == "sb3":
    print("Current working directory:", os.getcwd())
    if os.path.exists("td3_followCar.zip"):
        print("Model file found!")
    else:
        print("Model file missing!")

    model = sb3.TD3.load("td3_followCar_v0.zip") # Change if required 
    obs, _ = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)  
        obs, reward, done, truncated, info = env.step(action)

    env.close()  

if agentType == "agilerl":
    agent_path = "AgileRL_TD3_trained_agent.pt" # Change if required 

    if os.path.exists(agent_path):
        print("✅ Model file found!")
        print(f"File size: {os.path.getsize(agent_path)} bytes")
    else:
        print("❌ Model file not found!")

    # Load the trained agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_path = "AgileRL_TD3_trained_agent.pt"  # Make sure this matches your saved filename

    # Load the trained TD3 agent
    trained_agent = TD3.load(agent_path, device=device)

    # Set to evaluation mode
    trained_agent.eval()
    print("✅ TD3 Agent loaded successfully!")

    # Create environment (render mode to "human" for visualization)
    env = gym.make("followCar-v0", render_mode="human")

    # Reset the environment
    state, _ = env.reset()

    # Run the agent in the environment
    done = False
    while not done:
        action = trained_agent.get_action(state, training=False)  # Get action from trained agent
        next_state, reward, done, _, _ = env.step(action)  # Take action in env
        env.render()  # Render environment
        state = next_state  # Update state

    env.close()  # Close environment when done