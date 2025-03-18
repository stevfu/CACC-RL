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

On stable-baselines3 using TD3, after 3000 episodes on laptop 1650Ti-MaxQ: 

![followCar](trained_agent/td3_followCar_v1_AgileRL_1436/td3_followCar_v1_agileRL.gif)

## Changelog 
Version 0: Creation of initial designs and files 

Version 1: Heavy rework on environment, with updated kinematics, reward and training conditions

Version 1.1: Revised reward function, cleaned training data, and more visualization tools for training and testing values
