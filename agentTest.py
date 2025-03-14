import torch
import gymnasium as gym
import os
import imageio
from PIL import Image 

import gym_followCar

import agilerl
from agilerl.algorithms.td3 import TD3
from agilerl.utils.utils import make_vect_envs

import stable_baselines3 as sb3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

'''
Description:

Script to test trained agent on environments. Change <agentType> based on library used for training

'''


agentType = "sb3"

if agentType == "sb3":
    # Create environment (render mode as 'rgb_array' for saving frames)
    env = gym.make("followCar-v1", render_mode="rgb_array")

    # Load trained model
    model = sb3.TD3.load("trained_agent/td3_followCar_v1.zip")  # Change if required

    # Reset environment
    obs, _ = env.reset()
    done = False
    frames = []

    # Run one full episode
    while not done:
        action, _states = model.predict(obs, deterministic=True)  
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))  # Append only if valid
        
        # Stop after one episode
        if terminated or truncated:
            break  

    env.close()  # Close environment when done

    # Save as GIF
    gif_path = "trained_agent/td3_followCar_v1_2.gif"
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
        print(f"GIF saved at {gif_path}")
    else:
        print("No frames were captured. Check render settings.")

if agentType == "agilerl":
    env = gym.make("followCar-v0",render_mode = "human")
    agent_path = "AgileRL_TD3_trained_agent.pt" # Change if required 

    if os.path.exists(agent_path):
        print("Model file found!")
        print(f"File size: {os.path.getsize(agent_path)} bytes")
    else:
        print("Model file not found!")

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