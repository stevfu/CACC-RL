# followCarEnvironment-v1.py 

# Dependencies
import random
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np
import gymnasium as gym 
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
import json

class followCar_v1(gym.Env[np.ndarray, Union[int, np.ndarray]]): 
    '''
    ## Description: 
    The goal is to have a leader car moving in a forwards direction, with a follower car that keeps the same distance, 
    by accelerating and decelerating based on the leaders movements 

    Leader velocity is based upon NGSIM traffic data on the I-80 Emeryville interstate (https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data)
    Each epsisode, the leader velocity tracks a random vehicle and its velocity profile throughout the data collection

    ## Action Space : 
    The action is a normalized Box with a shape '(1,)' that can takes values in the set of {-1,1}. 
    that either tell the follow to:
    - action[0] < 0: deccelerate 
    - action[0] > 0: accelerate 

    ## Observation Space: 
    The observation will be a normalized Box with shape '(4,)' with the values corresponding to the following

    Number      Observation                 Min           Max
    0           Leader Position             0             positionThreshold 
    1           Leader Velocity (m/s)       0             33 (around 120km/h)
    2           Follower Position           0             positionThreshold
    3           Follower Velocity (m/s)     -33           33 (around 120km/h)

    ## Rewards 
    A robust reward function is used to ensure for proper training, where:
    (+) - Tracking Reward: reward for being within the following distance, which exponentially scales based on how close you are 
    (-) - Collision Penalty: Harsh penalty for when the follower car crashes into the leader car 
    (-) - Distance Penalty: Penalizes follower if they are too far from the leader car 
    (-) - Acceleration Penalty: Minor penalty for super harsh acceleration changes to prevent jerking 


    ## Starting State 
    The Leader Position is 25m ahead of the follower, while the Follower Position starts at 0 
    Leader Velocity will start at the first "frame_id" of the velocity profile
    Follower Velocity will start at 0 

    ## Episode End 
    The episode ends if: 
    - Termination: Distance greater than 50m
    - Termination: Both cars are touching 
    - Truncation: Reach Max Distance, velocity profile for following car finishes 

    ## Arguments 
    render_mode for gymnasium.make for pygame 

    '''

    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 60,
    }

    # This function is used to normalize the state variables in order to provide a more balanced range for the agent to train 
    def normalization(self, state): 
        leaderPosition, leaderVelocity, followerPosition, self.followerVelocity = state
        return np.array([leaderPosition/self.positionThreshold,
                         (leaderVelocity)/33,
                         followerPosition/self.positionThreshold,
                         self.followerVelocity/33], 
                         dtype=np.float32)
    
    ## Starting the environment 
    def __init__(self,render_mode: Optional[str] = None): 

        print("Using followCarEnvironment-v1\n")

        # Environment-specific parameters 
        self.tau = 0.1 # timestep
        self.time = 0
        # self.followingDistance = 60 # 60m required following distance between follower and leader 
        # self.distanceThreshold = 75 # 75m is the distance between the follower and leader for the episode to end 
        self.positionThreshold = 1000 # 1km in meters 
        self.timeheadway = 0

        # Initialize kinematic parameters 
        self.followerAcceleration = 0
        self.initialLeaderPosition = random.randint(75,125) # random start position to encourage better training 
        self.leaderVelocity = 0 
        self.followerVelocity = 0
        self.currentLeaderPosition = 0; 

        # Initialize rendering parameters 
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None 
        self.render_mode = render_mode

        # Set lower limits for observation space 
        lowerLimits = np.array( #minimum values 
            [
                0, #leader position
                0, #leader velocity, 33m/s correlates to 100km/hr 
                0, #follower position  
                0, #follower velocity, 33m/s correlates to 100km/hr 
            ],
            dtype = np.float32,
        )

        # Set upper limits for observation space 
        upperLimits = np.array( #maximum values 
            [ 
                self.positionThreshold, #leader position
                33,      #leader velocity, 33m/s correlates to 120km/hr
                self.positionThreshold, #follower position
                33,      #follower velocity, 33m/s correlates to 120km/hr 
            ],
            dtype = np.float32,
        )

        # Initialize action space and observation space 
        self.action_space = spaces.Box(low = -1, high = 1,shape = (1,), dtype = np.float32) # Normalized action space 
        self.observation_space = spaces.Box(lowerLimits, upperLimits, dtype = np.float32) 

        # Load from .json a dictionary of all vehicles (vehicle IDs) and their velocity profiles 
        with open("data/velocityProfiles.json","r") as f: 
            self.velocityProfiles = json.load(f) 

        # List of all the vehicles (vehicle IDs)
        self.unique_vehicle_ids = list(self.velocityProfiles.keys())
        
        # Pick a random vehicle as the first leader vehicle 
        self.vehicleID = self.unique_vehicle_ids[np.random.randint(0,len(self.unique_vehicle_ids))]
        self.leaderVelocityCounter = 0 # used to track the velocity profile 


    def step(self, action):     

        # Debugging action space 
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method"

        # Parameters to track
        self.time += self.tau # Episode Time 
        self.prevAcceleration = self.followerAcceleration # Acceleration in previous step 
        leaderPosition,self.leaderVelocity,followerPosition,self.followerVelocity = self.state # state variables 

        # Kinematics 
        self.leaderVelocity = self.velocityProfiles[self.vehicleID]["velocity"][self.leaderVelocityCounter] * 0.3048 # conversion from feet/sec to m/sec
        self.leaderVelocityCounter += 1
        leaderPosition += self.leaderVelocity * self.tau
        self.followerVelocity = np.clip(self.followerVelocity, -33, 33)
        self.followerAcceleration = action[0] * 3 # choose acceleration between -3m/s^2 to 3m/s^2
        self.followerVelocity += self.followerAcceleration * self.tau
        self.normalizedFollowerVelocity = max(0, self.followerVelocity) ## Prevent backwards movement 
        followerPosition += self.normalizedFollowerVelocity * self.tau + 0.5 * self.followerAcceleration * self.tau**2

        ## Lazy Implementation
        self.currentLeaderPosition = leaderPosition
        
        # Set state variables
        self.state = (leaderPosition,self.leaderVelocity,followerPosition,self.followerVelocity)

        #print(f"\nLeader Velocity: {self.leaderVelocity:.2f}")
        #print(f"Follower Velocity: {self.followerVelocity:.2f}")

        # Update time headway between cars 
        distanceHeadway = leaderPosition - followerPosition
        if (self.normalizedFollowerVelocity== 0):
            self.timeHeadway = float('inf') # Avoid division by zero 
        else: 
            self.timeHeadway = distanceHeadway / self.normalizedFollowerVelocity
        
        '''
        Reward function based on the follow factors: 
        - Proximity: Can the car keep a time headway that is close to 2.5s? 
        - Collision: If they collide, dock reward 
        - Smoothness: Can it accelerate and deccelerate in smooth intervals 
        '''

        max_timeHeadway = 15  # Cap for extreme cases
        normalized_timeHeadway = min(abs(self.timeHeadway), max_timeHeadway)
        reward = (
              (5) * np.exp(-((normalized_timeHeadway - 2.5) ** 2) / (3)) # Gaussian distribution proximity reward gives better encouragement 
            - (25) * (distanceHeadway <= 0 ) # collision
            - (10) * (abs(self.timeHeadway) > max_timeHeadway and distanceHeadway > 100) # too far away 
            - (0.1) * abs(self.prevAcceleration-self.followerAcceleration) # discourages large acceleration changes 
            - (1) * (self.followerVelocity < 0) # discourages going backwards 
        )

        # Termination:
        terminated = bool(
            distanceHeadway <= 0 or
            (abs(self.timeHeadway) > max_timeHeadway and distanceHeadway > 100)
        )
        
        # Truncation: 
        truncated = bool(
            self.leaderVelocityCounter >= len(self.velocityProfiles[self.vehicleID]["velocity"]) or
            leaderPosition > self.positionThreshold or 
            followerPosition > self.positionThreshold
        )

        # Check bounds to make sure we are still in, otherwise break 
        if terminated or truncated: 
            print(f"Total Episode Time: {self.time:.2f}")
            print(f"Total Reward: {reward:.2f}")
            if terminated: 
                print("Episode Terminated\n")
            else: 
                print("Episode Truncated\n")

        if self.render_mode == "human": 
            self.render()
        
        return self.normalization(self.state), reward, terminated, truncated, {}

    def reset(self, *,seed: Optional[int] = None, options: Optional[dict] = None,):

        super().reset(seed=seed)

        # Reset main environment parameters
        self.time = 0
        self.reward = 0
        self.leaderVelocityCounter = 0
        self.timeHeadway = 0

        # Generate new vehicleID
        self.vehicleID = self.unique_vehicle_ids[np.random.randint(0,len(self.unique_vehicle_ids))]
        
        # Same as initialization
        self.state = (self.initialLeaderPosition,self.leaderVelocity,self.initialLeaderPosition-25,np.random.uniform(0,15)) # state = (leaderPosition, leaderVelocity, followerPosition, self.followerVelocity)

        if self.render_mode == "human": 
            self.render() 

        print("Vehicle ID:", self.vehicleID)
        return self.normalization(self.state), {}
    
    def render(self): 

        # Error Handling 
        if self.render_mode is None: 
            assert self.spec is not None
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
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install pygame'
            ) from e 

        # Initialize pygame 
        if self.screen is None: 
            pygame.init() 
            if self.render_mode == "human": 
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("followCarEnvironment")
                font = pygame.font.Font(None,24)
            else: 
                self.screen = pygame.Surface ((self.screen_width, self.screen_height))

        if self.clock is None: 
            self.clock = pygame.time.Clock()
        
        # Initial Parameters
        font = pygame.font.Font(None, 36)

        world_width = self.positionThreshold + 50
        carHeight = 25
        carWidth = 50

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255,255,255))

        pygame.draw.line(self.surf, (0,0,0),(0,148),(self.screen_width,148),2)
    
        left, right, top, bottom = -carWidth, 0, 150, 150 + carHeight
        followerCoordinates = [(left, bottom) ,(left, top), (right, top), (right, bottom)]
        followerCoordinates = [(c[0] + x[2],c[1]) for c in followerCoordinates]
         
        gfxdraw.aapolygon(self.surf, followerCoordinates,(0,0,0))
        gfxdraw.filled_polygon(self.surf, followerCoordinates,(0,255,0))


        left, right, top, bottom = 0,carWidth, 150, 150 + carHeight
        leaderCoordinates = [(left, bottom) ,(left, top), (right, top), (right, bottom)]
        leaderCoordinates = [(l[0] + x[0],l[1]) for l in leaderCoordinates]

        gfxdraw.aapolygon(self.surf, leaderCoordinates,(0,0,0))
        gfxdraw.filled_polygon(self.surf, leaderCoordinates,(255,0,0))

        # Redraw Background 
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0,0))

        # Generate Text for Stats 
        displayVehicleID = "VehicleID: "  + str(int(self.vehicleID))
        textVehicleID = font.render(displayVehicleID, True, (0,0,0)) 
        textVehicleIDRect = textVehicleID.get_rect(topleft=(500, 25)) 

        displayLeaderVelocity = "Leader Velocity: "  + str(int(self.leaderVelocity)) +  " m/s"
        textLeaderVelocity = font.render(displayLeaderVelocity, True, (0, 0, 0))  
        textLeaderVelocityRect = textLeaderVelocity.get_rect(topleft=(500, 60))

        displayFollowerVelocity = "Follower Velocity: "  + str(int(self.followerVelocity)) + " m/s"
        textFollowerVelocity = font.render(displayFollowerVelocity, True, (0,0,0)) 
        textFollowerVelocityRect = textFollowerVelocity.get_rect(topleft=(500, 95)) 

        displayTimeHeadway = "Time Headway: "  + str(round(self.timeHeadway,1)) + " sec"
        textTimeHeadway = font.render(displayTimeHeadway, True, (0,0,0)) 
        textTimeHeadwayRect = textTimeHeadway.get_rect(topleft=(500, 130)) 

        displayLeaderPosition = "Leader Position: "  + str(round(self.currentLeaderPosition)) + " m"
        textLeaderPosition = font.render(displayLeaderPosition, True, (0,0,0)) 
        textLeaderPositionRect = textLeaderPosition.get_rect(topleft=(500, 165)) 

        # Render Texts
        self.screen.blit(textLeaderVelocity, textLeaderVelocityRect)  
        self.screen.blit(textFollowerVelocity, textFollowerVelocityRect)
        self.screen.blit(textVehicleID,textVehicleIDRect)
        self.screen.blit(textTimeHeadway,textTimeHeadwayRect)
        self.screen.blit(textLeaderPosition, textLeaderPositionRect)

        if self.render_mode == "human": 
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array": 
            return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.screen)), axes = (1,0,2)
            )
        
    def close(self): 
        if self.screen is not None: 
            import pygame 

            pygame.display.quit()
            pygame.quit()
            self.isopen = False 


