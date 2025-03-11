# CACC-RL

Research project under Dr. Ahmed Hamdi Sakr
 
** Work In Progress **

Implementation of reinforcement learning to train CACC controllers! 

Current followCarEnvironment has a "follower" car that tracks the motion of a "leader" car in a simple 1D environment 

The only input is the change of acceleration of the "follower" car

The leader car follow car velocity data from NGSIM datasets (https://datahub.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/). 

## Dependencies

With conda, use: 

`conda env create -f rl.yml`

## Usage 

Use followCarAgentTD3 to train agent with stable-baselines3, and use followCarAgileRL_TD3 to train agent with AgileRL. 

To view training results, use agentTest.py 

## Results

On stable-baselines3 using TD3, after ~5M training steps on laptop 1650Ti-MaxQ: 

![followCar]([https://github.com/stevfu/CACC-RL/blob/main/td3_followCar_v0.gif](https://github.com/stevfu/CACC-RL/blob/main/trained_agent/td3_followCar_v1_2.gif))

## Changelog 
Version 0: Creation of initial designs and files 
Version 1: Heavy rework on environment, with updated kinematics, reward and training conditions
