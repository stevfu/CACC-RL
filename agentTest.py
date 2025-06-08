# General dependecies 
import torch
import gymnasium as gym
import os
import imageio
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image 

# Environment
import gym_followCar
from gym_followCar.multiCarEnv import ParallelCarEnv
from pettingzoo.utils import ParallelEnv

# AgileRl dependencies 
import agilerl
from agilerl.algorithms.td3 import TD3 as agileTD3
from agilerl.utils.utils import make_vect_envs
from agilerl.algorithms.matd3 import MATD3  # Adjust import if needed

# Stable Baselines dependencies 
import stable_baselines3 as sb3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
 
'''
Description:

Script to test trained agent on environments. Change <agentType> based on library used for training

'''

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
    

# CHANGE BASED ON AGENT 
agentType = "agilerl"
agent = "multi"

if agent == "single":
    if agentType == "sb3":
        env = gym.make("followCar-v1", render_mode="rgb_array")
        model = sb3.TD3.load("trained_agent/td3_followCar_v1.zip")  # Change if required

        obs, _ = env.reset()
        done = False
        frames = []

        # Run one full episode
        while not done:
            action, _states = model.predict(obs, deterministic=True)  
            obs, reward, terminated, truncated, info = env.step(action)
            
            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))  # Append only if valid
            
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
        env = gym.make("followCar-v1", render_mode="rgb_array")
        agent_path = "trained_agent/agileRL_TD3_followCar_v3_1.pt"  # Change if required

        vehicle_id = ""; 
        leaderVelocities = []
        followerVelocities = [] 

        timeHeadwayProfile = [] 
        distanceHeadwayProfile = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trained_agent = agileTD3.load(agent_path, device=device)
        env = gym.make("followCar-v1", render_mode="rgb_array")

        state, _ = env.reset()
        frames = []

        # Run one episode
        done = False
        while not done:
            action = trained_agent.get_action(state, training=False)
            action = np.array(action)  # Convert to numpy array if not already
            action = action.squeeze() # Flatten the action to make sure it's in the correct form for env.step()
            if action.shape == ():
                action = np.array([action])  # Wrap it into an array if it's a scalar float

            next_state, reward, terminated, truncated, info = env.step(action)

            # Grab values to plot later 
            leader_v = next_state[1]*33
            follower_v = next_state[3]*33

            distanceHeadway = (next_state[0]-next_state[2])*1000 # Based on positionThreshold 
            timeHeadway = distanceHeadway / (next_state[3]*33)
            
            leaderVelocities.append(leader_v)
            followerVelocities.append(follower_v)

            timeHeadwayProfile.append(timeHeadway)
            distanceHeadwayProfile.append(distanceHeadway)

            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))  

            if terminated or truncated:
                vehicle_id = info.get("Vehicle ID",0)
                break  

            state = next_state

        env.close()

        # Create New Folder 
        folderPath = "trained_agent/td3_followCar_v1_AgileRL_" + str(vehicle_id)
        os.mkdir(folderPath)

        # Save as GIF
        gif_path = folderPath+"/td3_followCar_v1_agileRL.gif"
        if frames:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
            print(f"GIF saved at {gif_path}")
        else:
            print("No frames were captured. Check render settings.")

        # Plot follower vs. leader velocities 

        plot.figure(figsize=(10, 5))
        plot.plot(leaderVelocities, label="Leader Velocity", linestyle="-")
        plot.plot(followerVelocities, label="Follower Velocity", linestyle="--")
        plot.xlabel("Time Step")
        plot.ylabel("Velocity (m/s)")
        plot.title("Leader vs. Follower Velocities")
        plot.legend()
        plot.grid(True)
        #plot.show()
        plot.savefig(folderPath+"/Velocity")

        # Plot time headway 

        plot.figure(figsize=(10, 5))
        plot.plot(timeHeadwayProfile, label="Time Headway", linestyle="-")
        plot.xlabel("Time Step")
        plot.ylabel("Seconds (s) ")
        plot.title("Time Headway")
        plot.legend()
        plot.grid(True)
        #plot.show()
        plot.savefig(folderPath + "/Time Headway")

        # Plot Distance Headway 

        plot.figure(figsize=(10, 5))
        plot.plot(distanceHeadwayProfile, label="Distance Headway", linestyle="-")
        plot.xlabel("Time Step")
        plot.ylabel(" Distance (m)")
        plot.title("Distance Headway")
        plot.legend()
        plot.grid(True)
        #plot.show()
        plot.savefig(folderPath + "/Distance Headway")

if agent == "multi": 

    # --- Parameters ---
    n_followers = 3  # Set to match your training
    agent_path = "trained_agent/MATD3/multiCarAgent_3.pt"

    # --- Load environment ---
    env = ParallelCarEnv(n_followers=n_followers, render_mode="rgb_array")
    env = EarlyTerminationWrapper(env)  # Wrap to handle early termination
    obs, info = env.reset()
    frames = []

    # --- Load trained agent ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_agent = MATD3.load(agent_path, device=device)

    done = {agent: False for agent in env.agents}

    while not all(done.values()):
    # Get actions for all agents at once
        actions = trained_agent.get_action(obs, training=False)
        if isinstance(actions, tuple):
            actions = actions[0]  # Unpack if tuple returned
        actions = {k: np.array(v).squeeze() for k, v in actions.items()}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
            
        done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
       
    env.close()

    # Save as GIF
    gif_path = "trained_agent/MATD3/multiCarAgent_3_eval.gif"
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
        print(f"GIF saved at {gif_path}")
    else:
        print("No frames were captured. Check render settings.")



