import numpy as np
import random

from gymnasium import spaces
from gymnasium.utils import seeding
from pygame import gfxdraw

from pettingzoo.sisl._utils import Agent

class followerCars: 


    def __init__(self, initialLeaderPosition, initialLeaderVelocity):  
        self.initialLeaderPosition = initialLeaderPosition
        self.initialLeaderVelocity = initialLeaderVelocity

        self.position = 0
        self.velocity = 0

        self.state = self._initialize_state(initialLeaderPosition, initialLeaderVelocity)

    def _initialize_state(self, initialLeaderPosition, initialLeaderVelocity): 
        self.position = initialLeaderPosition-np.random.uniform(25,50) 
        self.velocity = np.random.uniform(initialLeaderVelocity-3,initialLeaderVelocity+3)
        
        self.state = (self.position, self.velocity)

        return self.state 
    
    def step(self, actions, tau): 
        self.acceleration = actions[0] * 3
        self.velocity += self.acceleration * tau
        self.velocity = np.clip(self.velocity,0,33) 
        self.position += self.velocity*self.tau+0.5*self.acceleration*self.tau**2

        self.state = (self.position, self.velocity)

    def reset(self, initialLeaderPosition, initialLeaderVelocity): 
        self.state = self._initialize_state(self, initialLeaderPosition, initialLeaderVelocity)

    def getPosition(self): 
        return self.position
    
    def getVelocity(self): 
        return self.velocity
    


class followCarsBase: 
    
    metadata = {
        "render_modes": ["human", "rbg_array"], 
        "render_fps": 60,
    }
    
    def __init__(self, n_followers = 3,render_mode = "human"): 
        self.n_followers = n_followers 

        # Environment-specific parameters
        self.time = 0
        self.tau = 0.1 
        self.positionThreshold = 1000
        self.cumulativeReward = 0

        # Initialize kinematic parameters 
        self.initialLeaderPosition = random.rand(75,125) # random start position to encourage better training 
        self.leaderVelocity = 0 
        self.currentLeaderPosition = self.initialLeaderPosition; 

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None 
        self.render_mode = render_mode

        lowerLimits = np.array( #minimum values 
            [
                0, #leader position
                0, #leader velocity, 33m/s correlates to 100km/hr 
                0, #follower position 
                0, #follower velocity, 33m/s correlates to 100km/hr 
            ],
            dtype = np.float32,
        )

        upperLimits = np.array( #maximum values 
            [ 
                self.positionThreshold, #leader position
                33,      #leader velocity, 33m/s correlates to 120km/hr
                self.positionThreshold, #follower position
                33,      #follower velocity, 33m/s correlates to 120km/hr 
            ],
            dtype = np.float32,
        )


        # Load from .json a dictionary of all vehicles (vehicle IDs) and their velocity profiles 
        with open("data/velocityProfiles.json","r") as f: 
            self.velocityProfiles = json.load(f) 

        # List of all the vehicles (vehicle IDs)
        self.unique_vehicle_ids = list(self.velocityProfiles.keys())
        
        # Pick a random vehicle as the first leader vehicle 
        self.vehicleID = self.unique_vehicle_ids[np.random.randint(0,len(self.unique_vehicle_ids))]
        self.leaderVelocityCounter = 0 # used to track the velocity profile 

        self.initialLeaderVelocity = self.velocityProfiles[self.vehicleID]["velocity"][self.leaderVelocityCounter] * 0.3048

        self.followers = [followerCars(self,self.initialLeaderPosition-np.random.uniform(25,50) - ((i+1) *50), np.random.uniform(self.leaderVelocity-3,self.leaderVelocity+3), self.tau ) for i in range(self.n_followers-1)]
        self.agents = [f"follower_{i}" for i in range(self.n_followers)]

        self.distanceHeadways = []
        self.timeHeadways = []

    
    def step(self, actions, seed=None, options=None,): 
        self.time += self.tau

        # Move Leader
        self.leaderVelocity = self.velocityProfiles[self.vehicleID]["velocity"][self.leaderVelocityCounter] * 0.3048 # conversion from feet/sec to m/sec
        self.leaderVelocityCounter += 1
        leaderPosition += self.leaderVelocity * self.tau
        self.currentLeaderPosition = leaderPosition

        # Move Follower 
        for i, follower in enumerate(self.followers):
            action = actions[f"follower_{i}"]
            follower.step(action, self.tau)

        # Headways 
        for i in range(len(self.followers) - 1):
            if i == 0: 
                distance = leaderPosition - self.followers[i].getPosition()
                time = distance/self.followers[i].getVelocity
            else: 
                distance = self.followers[i].getPosition() - self.followers[i+1].getPosition()
                time = distance/self.followers[i+1].getVelocity

            self.distanceHeadways[i] = distance

            max_timeHeadway = 15  # Cap for extreme cases
            normalized_time = min(abs(time), max_timeHeadway) ## fix 
            self.timeHeadways[i] = normalized_time

        # Rewards
        for i in range (len(self.followers)-1): 
            x = max(1e-6, abs(self.timeHeadways[i]))
            mew = 0.4226 
            sigma = 0.4365
            reward = ( 
                (10) * (1/(x*sigma*np.sqrt(2*np.pi)))*np.exp(-((np.log(x)-mew)**2)/(2*(sigma**2))) ## Log normal probability distribution proximity reward 
                - (25) * (self.distanceHeadways[i] <= 0 ) # collision
                - (10) * (abs(self.timeHeadway) > max_timeHeadway and self.distanceHeadways[i] > 100) # too far away 
                - (0.1) * abs(self.prevAcceleration-self.followerAcceleration) # discourages large acceleration changes 
                - (100) * (self.followerVelocity < 0) # discourages going backwards 
            )
            self.cumulativeReward += reward ## fix 

        
        # Termination
        terminated = {agent: False for agent in self.agents}

        for i in range(len(self.followers) - 1):
            if self.distanceHeadways[i] <= 0: 
                terminated[f"follower_{i + 1}"] = True
            if self.timeHeadways[i] > max_timeHeadway and self.distanceHeadways[i] > 100: ## remove 
                terminated[f"follower_{i+1}"] = True
        
        # Truncation 
        truncated = bool(
            self.leaderVelocityCounter >= len(self.velocityProfiles[self.vehicleID]["velocity"]) or
            leaderPosition > self.positionThreshold
        ) 

        if self.render_mode == "human": 
            self.render() 
        
        observations = {self.agents[i]: (
            self.followers[i].getPosition, 
            self.follower[i].getVelocity, 
            self.distanceHeadways[i], 
            self.timeHeadways[i]
            
        ) for i in range(len(self.followers)-1)}

        return observations, reward, terminated, truncated, {}, {}

        
    def reset(self,seed):

        super.reset(seed=seed)

        self.time = 0 
        self.cumulativeReward = 0
        self.leaderVelocityCounter = 0
        self.currentLeaderPosition = self.initialLeaderPosition
        self.distanceHeadways = [0,0,0]
        self.timeHeadways = [0,0,0]

        self.leaderVelocity = self.velocityProfiles[self.vehicleID]["velocity"][self.leaderVelocityCounter] * 0.3048

        self.vehicleID = self.unique_vehicle_ids[np.random.randint(0,len(self.unique_vehicle_ids))]

        self.followers = [followerCars(self,self.initialLeaderPosition, self.initialLeaderVelocity, self.tau ) for _ in range(self.n_followers)]
        self.agents = [f"follower_{i}" for i in range(self.n_followers)]

        observations = {self.agents[i]: (
            self.followers[i].getPosition, 
            self.follower[i].getVelocity, 
            self.distanceHeadways[i], 
            self.timeHeadways[i]
            
        ) for i in range(len(self.followers)-1)}

        infos = {a: {} for a in self.agents}

        return observations, infos
    
    def render(self): 
        # Error Handling 
        if self.render_mode is None: 
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

        offset = 100

        x = [self.currentLeaderPosition, self.followers[0].getPosition, self.followers[1].getPosition, self.followers[2].getPosition]
        
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255,255,255))

        pygame.draw.line(self.surf, (0,0,0),(0,148),(self.screen_width,148),2)

        # leader

        left, right, top, bottom = -carWidth, 0, 150, 150 + carHeight
        leaderCoordinates = [(left, bottom) ,(left, top), (right, top), (right, bottom)]
        leaderCoordinates = [(c[0] + x[0] + offset ,c[1]) for c in leaderCoordinates]
         
        gfxdraw.aapolygon(self.surf, followerCoordinates_3,(0,0,0))
        gfxdraw.filled_polygon(self.surf, followerCoordinates_3,(0,255,0))

        # first follower
        left, right, top, bottom = -carWidth, 0, 150, 150 + carHeight
        followerCoordinates_1 = [(left, bottom) ,(left, top), (right, top), (right, bottom)]
        followerCoordinates_1 = [(c[0] + x[1] + offset ,c[1]) for c in followerCoordinates_1]
         
        gfxdraw.aapolygon(self.surf, followerCoordinates_1,(0,0,0))
        gfxdraw.filled_polygon(self.surf, followerCoordinates_1,(0,255,0))

        # second follower 

        left, right, top, bottom = -carWidth, 0, 150, 150 + carHeight
        followerCoordinates_2 = [(left, bottom) ,(left, top), (right, top), (right, bottom)]
        followerCoordinates_2 = [(c[0] + x[2] + offset ,c[1]) for c in followerCoordinates_2]
         
        gfxdraw.aapolygon(self.surf, followerCoordinates_2,(0,0,0))
        gfxdraw.filled_polygon(self.surf, followerCoordinates_2,(0,255,0))

        # third follower

        left, right, top, bottom = -carWidth, 0, 150, 150 + carHeight
        followerCoordinates_3 = [(left, bottom) ,(left, top), (right, top), (right, bottom)]
        followerCoordinates_3 = [(c[0] + x[3] + offset ,c[1]) for c in followerCoordinates_3]
         
        gfxdraw.aapolygon(self.surf, followerCoordinates_3,(0,0,0))
        gfxdraw.filled_polygon(self.surf, followerCoordinates_3,(0,255,0))

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

        displayLeaderPosition = "Leader Position: "  + str(round(self.currentLeaderPosition)) + " m"
        textLeaderPosition = font.render(displayLeaderPosition, True, (0,0,0)) 
        textLeaderPositionRect = textLeaderPosition.get_rect(topleft=(500, 165)) 

        # Render Texts
        self.screen.blit(textLeaderVelocity, textLeaderVelocityRect)  
        self.screen.blit(textVehicleID,textVehicleIDRect)
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








        



        