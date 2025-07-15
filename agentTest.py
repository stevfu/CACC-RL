# General dependecies 
import torch
import gymnasium as gym
import os
import imageio
import numpy as np
import pandas as pd 
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
    agent_path = "trained_agent/MATD3/20250715_1_multi.pt"

    # --- Load environment ---
    env = ParallelCarEnv(n_followers=n_followers, render_mode="debug")
    obs, info = env.reset()
    frames = []

    # --- Load trained agent ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_agent = MATD3.load(agent_path, device=device)

    # --- Automated output naming ---
    base_name = os.path.splitext(os.path.basename(agent_path))[0]  # e.g. '20250713_multi'
    output_dir = os.path.dirname(agent_path)
    excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
    gif_path = os.path.join(output_dir, f"{base_name}.gif")
    plot_dir = os.path.join(output_dir, base_name)
    os.makedirs(plot_dir, exist_ok=True)

    done = {agent: False for agent in env.agents}

    car_positions = {agent: [] for agent in env.agents}
    car_velocities = {agent: [] for agent in env.agents}
    car_accelerations = {agent: [] for agent in env.agents}
    car_rewards = {agent: [] for agent in env.agents}
    # Add leader tracking
    leader_positions = []
    leader_velocities = []

    # Prepare to track reward components per agent per step
    reward_components = {agent: {key: [] for key in ["lognorm", "forward", "ttc", "collision", "close", "jerk", "reverse", "gap"]} for agent in env.agents}

    while not all(done.values()):
        # Get actions for all agents at once
        actions = trained_agent.get_action(obs, training=False)
        if isinstance(actions, tuple):
            actions = actions[0]  # Unpack if tuple returned
        actions = {k: np.array(v).squeeze() for k, v in actions.items()}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        print("Rewards:", rewards)
        print("\nObs:", obs)

        for i, agent in enumerate(env.agents):
            car_positions[agent].append(env.followers[i].position)
            car_velocities[agent].append(env.followers[i].velocity)
            car_accelerations[agent].append(env.followers[i].acceleration)
            car_rewards[agent].append(rewards[agent])

            # --- Use reward breakdowns from infos ---
            for key in reward_components[agent].keys():
                reward_components[agent][key].append(infos[agent].get(key, np.nan))


        leader_positions.append(env.leader_position)
        leader_velocities.append(env.leader_velocity)
        

        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
            
        done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
       
    env.close()

    # Save Results 
    # Prepare the data for export
    reward_df = pd.DataFrame({agent: car_rewards[agent] for agent in env.agents})

    # Ensure output directory exists before saving Excel files
    os.makedirs("trained_agent/MATD3", exist_ok=True)

    # Prepare cumulative reward DataFrame for the first sheet
    cumulative_reward_df = pd.DataFrame({agent: np.cumsum(car_rewards[agent]) for agent in env.agents})

    # Save all sheets in one go for efficiency
    # Prepare position and velocity DataFrames
    position_df = pd.DataFrame({agent: car_positions[agent] for agent in env.agents})
    velocity_df = pd.DataFrame({agent: car_velocities[agent] for agent in env.agents})

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        cumulative_reward_df.to_excel(writer, sheet_name="TotalReward", index_label="Time Step")
        position_df.to_excel(writer, sheet_name="Positions", index_label="Time Step")
        velocity_df.to_excel(writer, sheet_name="Velocities", index_label="Time Step")
        for agent in env.agents:
            breakdown_df = pd.DataFrame({
                "lognorm": reward_components[agent]["lognorm"],
                "forward": reward_components[agent]["forward"],
                "ttc": reward_components[agent]["ttc"],
                "collision": reward_components[agent]["collision"],
                "close": reward_components[agent]["close"],
                "jerk": reward_components[agent]["jerk"],
                "reverse": reward_components[agent]["reverse"],
                "gap": reward_components[agent]["gap"],
            })
            breakdown_df.to_excel(writer, sheet_name=f"{agent}_breakdown", index_label="Time Step")

    # Save as GIF
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
        print(f"GIF saved at {gif_path}")
    else:
        print("No frames were captured. Check render settings.")

    # Position plot
    plot.figure(figsize=(10, 5))
    plot.plot(leader_positions, label="Leader Position", linestyle="-")
    for agent in env.agents:
        plot.plot(car_positions[agent], label=f"{agent} Position")
    plot.xlabel("Time Step")
    plot.ylabel("Position (m)")
    plot.title("Car Positions Across Episode")
    plot.legend()
    plot.grid(True)
    plot.show(block=False)
    plot.savefig(os.path.join(plot_dir, "positions.png"))

    # Velocity plot
    plot.figure(figsize=(10, 5))
    plot.plot(leader_velocities, label="Leader Velocity", linestyle="-")
    for agent in env.agents:
        plot.plot(car_velocities[agent], label=f"{agent} Velocity")
    plot.xlabel("Time Step")
    plot.ylabel("Velocity (m/s)")
    plot.title("Car Velocities Across Episode")
    plot.legend()
    plot.grid(True)
    plot.show(block=False)
    plot.savefig(os.path.join(plot_dir, "velocities.png"))

    # Acceleration plot
    plot.figure(figsize=(10, 5))
    for agent in env.agents:
        plot.plot(car_accelerations[agent], label=f"{agent} Acceleration")
    plot.xlabel("Time Step")
    plot.ylabel("Acceleration (m/s²)")
    plot.title("Car Acceleration")
    plot.legend()
    plot.grid(True)
    plot.show(block=False)
    plot.savefig(os.path.join(plot_dir, "accelerations.png"))

    # Cumulative Reward plot
    plot.figure(figsize=(10, 5))
    for agent in env.agents:
        cumulative_rewards = np.cumsum(car_rewards[agent])
        plot.plot(cumulative_rewards, label=f"{agent} Cumulative Reward")
    plot.xlabel("Time Step")
    plot.ylabel("Cumulative Reward")
    plot.title("Car Cumulative Rewards Across Episode")
    plot.legend()
    plot.grid(True)
    plot.show(block=False)
    plot.savefig(os.path.join(plot_dir, "cumulative_rewards.png"))



