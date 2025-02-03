import gymnasium as gym
import numpy as np

import gym_followCar

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

'''
Description: 

Use of stable-baselines3 library to train agents based on TD3 algorithm. 

'''

env = gym.make("followCar-v0", render_mode="human")

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,learning_rate=3e-4)
model.learn(total_timesteps=10000000, log_interval=10)
model.save("td3_followCar")
