# followCarEnvironment.py 
# Dependencies

import math
import random
from typing import Optional, Tuple, Union
import numpy as np
import gymnasium as gym 
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.envs.registration import register
from gymnasium.spaces import Box


class followCar(gym.Env[np.ndarray, Union[int, np.ndarray]]): 
    '''
    ## Description: 
    The goal is to have a leader car moving in one direction, accelerating and decelerating randomly, 
    with a follower car that keeps the same distance, by accelerating and decelerating based on the leaders movements 

    ## Action Space : 
    The action is a Box with a shape '(1,)' that can takes values in the set of {-5,5}
    that either tell the follow to:
    - action[0] < 0: deccelerate 
    - action[0] > 0: accelerate 

    ## Observation Space: 
    The observation will be a 'ndarray' with shape '(4,)' with the values corresponding to the following

    Number      Observation                 Min           Max
    0           Leader Position             0             positionThreshold   
    1           Leader Velocity (m/s)       3             5
    2           Follower Position           0             positionThreshold 
    3           Follower Velocity (m/s)     -5            5

    ## Rewards 
    A robust reward function is used to ensure for proper training, where:
    (+) - Tracking Reward: reward for being within the following distance, which exponentially scales based on how close you are 
    (-) - Collision Penalty: Harsh penalty for when the follower car crashes into the leader car 
    (-) - Distance Penalty: Penalizes follower if they are too far from the leader car 
    (-) - Acceleration Penalty: Minor penalty for super harsh acceleration changes to prevent jerking 


    ## Starting State 
    The Leader Position is 25m ahead of the follower, while the Follower Position starts at 0 
    Leader Velocity and Follower velocity will start at 0 

    ## Episode End 
    The episode ends if: 
    - Termination: Distance greater than 50m
    - Termination: Both cars are touching 
    - Truncation: Reach Max Distance 

    ## Arguments 
    render_mode for gymnasium.make for pygame 

    '''

    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 60,
    }

    # This function is used to normalize the state variables in order to provide a more balanced range for the agent to train 
    def normalization(self, state): 
        leaderPosition, leaderVelocity, followerPosition, followerVelocity = state
        return np.array([leaderPosition/self.positionThreshold,
                         (leaderVelocity-3)/2,
                         followerPosition/self.positionThreshold,
                         followerVelocity/5], dtype=np.float32)
    

    def __init__(self,render_mode: Optional[str] = None): 

        self.tau = 0.1 # timestep

        self.time = 0

        self.followingDistance = 15 # required following distance between follower and leader 

        self.distanceThreshold = 50 # the distance between the follower and leader for the episode to end 

        self.followerAcceleration = 0

        self.positionThreshold = 600

        self.initialLeaderPosition = 100

        self.leaderVelocity=0

        
        lowerLimits = np.array( #minimum values 
            [
                0, # leader position
                3, #leader velocity 
                0, #follower position  
                -3, #follower velocity 
            ],
            dtype = np.float32,
        )

        upperLimits = np.array( #maximum values 
            [ 
                self.positionThreshold, #leader position
                5,      #leader velocity 
                self.positionThreshold, #follower position
                3,      #follower velocity 
            ],
            dtype = np.float32,
        )

        # Initialize action space and observation space 

        # self.action_space = spaces.Discrete(10)
        self.action_space = spaces.Box(low = -5, high = 5,shape = (1,), dtype = np.float32)
    
        self.observation_space = spaces.Box(lowerLimits, upperLimits, dtype = np.float32)

        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None 
        self.render_mode = render_mode
        self.steps_beyond_terminated = None 

    def step(self, action):     

        ## Keep track of episode time 
        self.time += self.tau

        ## Keep track of previous acceleration 
        self.prevAcceleration = self.followerAcceleration
                

        # Debugging action space 
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method"

        # Initial Variables 
        leaderPosition,leaderVelocity,followerPosition,followerVelocity = self.state 
        self.leaderVelocity = np.sin(self.time * 2 * np.pi) + 4  #add offset
        
        # Updating Position and Velocity 
        leaderPosition += leaderVelocity * self.tau

        self.followerAcceleration = action[0]

        followerVelocity += self.followerAcceleration * self.tau
        followerPosition += followerVelocity * self.tau + 0.5 * self.followerAcceleration * self.tau**2

        self.state = (leaderPosition,leaderVelocity,followerPosition,followerVelocity)

        # Update distance between the two cars 
        relativePosition = leaderPosition - followerPosition
            
        # Check for state conditions (positive and negative) 
      
        '''
        Reward function based on the follow factors: 
        - Proximity: Can the car keep the same distance from the leader 
        - Collision: If they collide, dock reward 
        - Smoothness: Can it accelerate and deccelerate in smooth intervals 
        '''
        reward = (
            (10) * np.exp(-0.1 * abs(relativePosition- self.followingDistance)) # non-linear proximity reward gives better encouragement 
            - (100) * (relativePosition < 0 ) # collision
            - (50) * max(0,relativePosition-self.distanceThreshold) # being outside of following distance 
            - (0.1) * abs(self.prevAcceleration-self.followerAcceleration) # discourages large acceleration changes 
        )


    
        # Conditions for agent to avoid such as the two cars "crashing" or being too far apart 
        terminated = bool(
            relativePosition >= self.distanceThreshold  or  #too far 
            relativePosition < 0 or 
            leaderPosition > self.screen_width or 
            followerPosition > self.screen_width  # too close 
        )

         # Check bounds to make sure we are still in, otherwise break 
        if terminated: 
            print(f"\nTotal Episode Time: {self.time:.2f}")
            print(f"\nTotal Reward: {reward:.2f}")

        if self.render_mode == "human": 
            self.render()
        
        
        return self.normalization(self.state), reward, terminated, False, {}

    def reset(self, *,seed: Optional[int] = None, options: Optional[dict] = None,):
        
        super().reset(seed=seed)

        self.steps_beyond_terminated = None

        self.time = 0

        self.reward = 0

        # same as initialization 
        self.state = (self.initialLeaderPosition,0,self.initialLeaderPosition-25,0) ## state = (leaderPosition, leaderVelocity, followerPosition, followerVelocity)

        if self.render_mode == "human": 
            self.render() 

        return self.normalization(self.state), {}
    

    def render(self): 
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

        if self.screen is None: 
            pygame.init() 
            if self.render_mode == "human": 
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("followCarEnvironment")
            else: 
                self.screen = pygame.Surface ((self.screen_width, self.screen_height))


        if self.clock is None: 
            self.clock = pygame.time.Clock()
        
        world_width = self.positionThreshold +50
        scale = self.screen_width / world_width 
        carHeight = 25
        carWidth = 50

        font = pygame.font.Font(None, 36)


        if self.state is None: 
            return None 
        
        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255,255,255))
    
       

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

        displayText = "Leader Velocity: "  + str(self.leaderVelocity)
        text_surface = font.render(displayText, True, (255, 0, 0))  # White text
        text_rect = text_surface.get_rect(center=(650, 150))  # Centered position

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0,0))

        self.screen.blit(text_surface, text_rect)  # Draw text onto screen

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


