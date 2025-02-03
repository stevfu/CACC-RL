# CACC-RL

Research project under Dr. Ahmed Hamdi Sakr
 
* Work In Progress

Implementation of reinforcement learning to train CACC controllers! 

Current followCarEnvironment has a "follower" car that tracks the motion of a "leader" car in a simple 1D environment 

The only input is the change of acceleration of the "follower" car

## Dependencies

With conda, use: 

`conda env create -f rl.yml`

## Usage 

Use followCarAgentTD3 to train agent with stable-baselines3, and use followCarAgileRL_TD3 to train agent with AgileRL. 

To view training results, use agentTest.py 

## Results

On stable-baselines3 using TD3, after ~5M training steps on laptop 1650Ti-MaxQ: 

![followCar](https://github.com/stevfu/CACC-RL/blob/main/td3_followCar_v0.gif)

## Changelog 
Version 1.0: Creation of inital designs and files 
