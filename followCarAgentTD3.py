from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
import gym_followCar
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import torch 

class EpisodeLimitCallback(BaseCallback):
    def __init__(self, max_episodes, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if the episode has ended
        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_count += 1
            print(f"Episode {self.episode_count} finished")
            if self.verbose:
                print(f"Episode {self.episode_count} finished")
            # Stop training if max episodes are reached
            if self.episode_count >= self.max_episodes:
                print("Reached episode limit. Stopping training.")
                return False
        return True

# Initialize environment
env = gym.make("followCar", render_mode="human")

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize model
print(torch.cuda.is_available())  # Should print True
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=1e-4,device="cuda")

# Train with episode limit
episode_callback = EpisodeLimitCallback(max_episodes=6000)  # Set your episode limit
model.learn(total_timesteps=int(1e6), log_interval=10, callback=episode_callback)

# Save model
model.save("td3_followCar_v1")