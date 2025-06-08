import numpy as np
import json
import random
import pygame

from pygame import gfxdraw
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn




class Car:

    
    # Initializes position and velocity of car 
    def __init__(self, initial_position, initial_velocity): 
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.previous_acceleration = 0
        self.reset()
    
    # Reset car state 
    def reset(self):
        self.position = self.initial_position
        self.velocity = self.initial_velocity
        self.acceleration = 0.0

    # Step in environment 
    def step(self, action, tau):
        action = np.array(action).flatten()
        self.previous_acceleration = self.acceleration
        self.acceleration = action[0] * 3
        self.velocity = np.clip(self.velocity + self.acceleration * tau, 0, 33)
        self.position += self.velocity * tau + 0.5 * self.acceleration * tau ** 2

    # Return state 
    def get_state(self):
        return np.array([self.position, self.velocity], dtype=np.float32)


class ParallelCarEnv(ParallelEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Initializes the environment 
    def __init__(self, n_followers=3, render_mode="human"): 
        super().__init__()

        # Initial parameters 
        self.n_followers = n_followers
        self.render_mode = render_mode
        self.tau = 0.1
        self.position_threshold = 1000
        self.screen = None
        self.clock = None
        self.car_length = 5 # Length of the car in meters

        # Load velocity profiles
        with open("data/velocityProfiles.json", "r") as f:
            self.velocity_profiles = json.load(f)
        self.unique_vehicle_ids = list(self.velocity_profiles.keys())

        # Spaces
        obs_low = np.array([0, 0, 0,0], dtype=np.float32)
        obs_high = np.array([self.position_threshold, 33, self.position_threshold, self.position_threshold], dtype=np.float32)
        self.observation_spaces = {
            f"follower_{i}": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            for i in range(self.n_followers)
        }
        self.action_spaces = {
            f"follower_{i}": spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32
            )
            for i in range(self.n_followers)
        }

        # Generate agents (follower cars) 
        self.agents = [f"follower_{i}" for i in range(self.n_followers)]
        self.possible_agents = self.agents.copy()

    def reset(self, seed=None, options=None):
        # Initial parameters 
        self.time = 0
        self.leader_velocity_counter = 0
        self.vehicle_id = random.choice(self.unique_vehicle_ids)
        self.leader_velocity = self.velocity_profiles[self.vehicle_id]["velocity"][0] * 0.3048
        self.leader_position = random.uniform(250,300)

        # Reset followers 
        if hasattr(self, "followers") and len(self.followers) == self.n_followers:
            # If followers already exist, reset their positions and velocities
            for i, follower in enumerate(self.followers):
                follower.initial_position = self.leader_position - 50 * (i + 1) - random.uniform(-10, 10)
                follower.initial_velocity = np.clip(self.leader_velocity + random.uniform(-3, 3), 0, 33)
                follower.reset()
        else:
            # Create new followers if they don't exist 
            self.followers = [
                Car(
                    initial_position=self.leader_position - 50 * (i + 1) - random.uniform(-10, 10),
                    initial_velocity=np.clip(self.leader_velocity + random.uniform(-3, 3), 0, 33)
                ) for i in range(self.n_followers)
    ]

        # Reset all reward parameters 
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        observations = self._get_observations()
        return observations, {}

    def step(self, actions):
        # Move environment foward 
        self.time += self.tau

        # Update leader
        self.leader_velocity = self.velocity_profiles[self.vehicle_id]["velocity"][self.leader_velocity_counter] * 0.3048
        self.leader_velocity_counter += 1
        self.leader_position += self.leader_velocity * self.tau

        # Update followers
        for i, agent in enumerate(self.agents):
            self.followers[i].step(actions[agent], self.tau)

        # Compute rewards, terminations, truncations
        max_timeHeadway = 15  # Cap for extreme cases
        #[ follower[2] --> follower[1] --> follower[0] --> leader ]
        for i, agent in enumerate(self.agents):
            if i == 0: # if it is the follower car right behind the leader 
                front_position = self.leader_position
                velocity = self.leader_velocity
            else: # if it is a follower car right behind another follower car 
                front_position = self.followers[i-1].position
                velocity = self.followers[i-1].velocity
            distanceHeadway = front_position - self.followers[i].position
            timeHeadway = distanceHeadway / (velocity + 1e-6) 
            relativeVelocity = velocity - self.followers[i].velocity

            # Time to collision
            ttc_threshold = 4  # seconds
            if relativeVelocity > 0:  # If follower is slower than the car in front
                timeToCollision = distanceHeadway / relativeVelocity
                if 0 < timeToCollision <= ttc_threshold: 
                    ttcPenalty = np.log(timeToCollision/ttc_threshold); 
                else: 
                    ttcPenalty = 0
            
            else: 
                ttcPenalty = 0

            # Reward based on log-normal distribution (encouraging time headway ~ 1.5s)
            mew = 0.4226
            sigma = 0.4365
            x = max(1e-6, abs(timeHeadway))
            reward = (
              (10) * (1/(x*sigma*np.sqrt(2*np.pi)))*np.exp(-((np.log(x)-mew)**2)/(2*(sigma**2))) ## Log normal probability distribution proximity reward 
            + (10) * ttcPenalty
            - (250) * (distanceHeadway <= self.car_length ) # collision
            #- (10) * (abs(timeHeadway) > max_timeHeadway and distanceHeadway > 100) # too far away 
            - (1) * abs(self.followers[i].previous_acceleration-self.followers[i].acceleration) # discourages large acceleration changes 
            - (100) * (self.followers[i].velocity < 0) # discourages going backwards 
            )
            self.rewards[agent] = reward #each agent gets its own reward 

            # Termination 
            if distanceHeadway <= self.car_length:
            #or abs(timeHeadway) > max_timeHeadway and distanceHeadway > 100: 
                self.terminations[agent] = True
                #print("\nRewards: ", self.rewards)
            else:   
                self.terminations[agent] = False
            
        # Global Truncation
        for agent in self.agents:
            if self.leader_velocity_counter >= len(self.velocity_profiles[self.vehicle_id]["velocity"]) or self.leader_position >= self.position_threshold:
                self.truncations[agent] = True
                #print("\nRewards: ", self.rewards)
            else: 
                self.truncations[agent] = False 

        # Render environment 
        if self.render_mode == "human":
            self.render()

        #print("Terminations: ", self.terminations)
        #print("Truncations: ", self.truncations)
        return self._get_observations(), self.rewards, self.terminations, self.truncations, self.infos
    

    def _get_observations(self):
        observations = {}
        for i, agent in enumerate(self.agents):
            front_position = self.leader_position if i == 0 else self.followers[i - 1].position
            state = self.followers[i].get_state()
            distanceHeadway = front_position - self.followers[i].position
            observations[agent] = np.array([state[0]/self.position_threshold, 
                                            state[1]/33, 
                                            distanceHeadway/self.position_threshold, 
                                            front_position/self.position_threshold], 
                                            dtype=np.float32) # normalized state 
        return observations
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode is None:
            assert hasattr(self, "spec") and self.spec is not None
            import gymnasium as gym
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
                "You can specify the render_mode at initialization,"
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise ImportError('pygame is not installed, run `pip install pygame`') from e

        # Initialize pygame
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((800, 600))
                pygame.display.set_caption("multiCarEnv")
            else:
                self.screen = pygame.Surface((800, 600))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        font = pygame.font.Font(None, 24)
        world_width = self.position_threshold + 50
        car_height = 25
        car_width = 50
        road_y = 400 

        # Draw background
        surf = pygame.Surface((800, 600))
        surf.fill((255, 255, 255))
        pygame.draw.line(surf, (0, 0, 0), (0, road_y), (800, road_y), 2)

        meters_to_pixels = 800 / self.position_threshold
        car_width = int(self.car_length * meters_to_pixels)

        # Draw leader car (red)
        leader_x = int(self.leader_position * meters_to_pixels)
        leader_coords = [
            (leader_x - car_width, road_y - car_height), (leader_x - car_width, road_y - 2 * car_height),
            (leader_x, road_y - 2 * car_height), (leader_x, road_y - car_height)
        ]
        gfxdraw.aapolygon(surf, leader_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, leader_coords, (255, 0, 0))

        # Draw followers (green)
        for i, follower in enumerate(self.followers):
            follower_x = int(follower.position * meters_to_pixels)
            follower_coords = [
                (follower_x - car_width, road_y - car_height), (follower_x - car_width, road_y - 2 * car_height),
                (follower_x, road_y - 2 * car_height), (follower_x, road_y - car_height)
            ]
            gfxdraw.aapolygon(surf, follower_coords, (0, 0, 0))
            gfxdraw.filled_polygon(surf, follower_coords, (0, 255, 0))

            # Draw follower stats
            display_follower_velocity = f"Follower {i} Velocity: {int(follower.velocity)} m/s"
            text_follower_velocity = font.render(display_follower_velocity, True, (0, 0, 0))
            surf.blit(text_follower_velocity, (500, 95 + 35 * i))

        # Draw leader stats
        display_leader_velocity = f"Leader Velocity: {int(self.leader_velocity)} m/s"
        text_leader_velocity = font.render(display_leader_velocity, True, (0, 0, 0))
        surf.blit(text_leader_velocity, (500, 25))

        display_leader_position = f"Leader Position: {int(self.leader_position)} m"
        text_leader_position = font.render(display_leader_position, True, (0, 0, 0))
        surf.blit(text_leader_position, (500, 60))

        # Flip and blit
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise KeyboardInterrupt  # This will stop training
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


def parallel_env():
    return ParallelCarEnv()

# Optional wrapper for compatibility
env = parallel_wrapper_fn(parallel_env)