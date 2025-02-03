import gymnasium as gym

import time 

import followCarEnvironment

from stable_baselines3 import DQN

env = gym.make("followCar-v0", render_mode="human")

model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001)
#model.learn(total_timesteps=3000000, log_interval=4)
#model.save("dqn_followCar")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_followCar")

obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
