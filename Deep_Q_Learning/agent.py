import torch
import gymnasium as gym
from torch import nn
from collections import deque
import numpy as np
import os
import device

# Here we are going to create a an agent designed for deep $Q$-neural networks.

class Agent_Q_fun(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q_function = nn.Sequential(
            nn.Conv2d(state_shape[2],16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Flatten(),
            nn.Linear(16*state_shape[0]*state_shape[1],32),
            nn.ReLU(),
            nn.Linear(32,n_actions),
            nn.ReLU()
        )


    def forward(self,x):
        x = self.normalize(x)
        x = self.Q_function(x)
        return x
    
    def normalize(self,state):
        max_c = 89
        min_c = 0

        state = (state - min_c)/(max_c - min_c + 1e8)
        return state
    
    def copy(self):
        copy = Agent_Q_fun(self.state_shape,self.n_actions)
        copy.load_state_dict(self.state_dict())
        return copy
    

class Agent:
    def __init__(self,env_name,Q_fun):
        """Initalize the base agent class, put the enviroment name for the atari game,
        and the Neural network used to approximate the Q-function"""
        # enviroment name, like ALE/SpaceInvaders-v5 This needs to be set up 
        self.env_name = env_name
        # make the enviroment
        self.env = gym.make(self.env_name,continuous=False)
        self.state, self.info = self.env.reset()


        # get the state shapes and 
        self.state_shape = self.state.shape
        self.n_actions = self.env.action_space.n
        #Define Q_function neural network
        self.Q_fun = Q_fun(self.state_shape,self.n_actions)
        self.Q_fun.to(device.DEVICE)
        self.env.close()

    
    def add_wrappers(self, env):
        return env

    def get_action(self,env,state,epsilon=0):
        
        if np.random.random()<epsilon:
            return int(env.action_space.sample())
        else:
            
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32,device=device.DEVICE)
            state_tensor = state_tensor.permute((0,3,1,2))
            return int(self.Q_fun(state_tensor).argmax().item())
        
    def get_replay(self,env,state,epsilon):

        action = self.get_action(env,state,epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        return state, action, next_state, reward, terminated
    
    def run_agent(self):
        with torch.no_grad():
            # make enviroment for testing
            env = gym.make(self.env_name,render_mode='human',obs_type='rgb',continuous=False)

            env = self.add_wrappers(env)

            obs, info = env.reset()

            done = False

            cum_reward = 0
            # Play
            while not done:
                
                action = self.get_action(env,obs,epsilon=0)
            
                next_obs, reward, terminated, truncated, info = env.step(action)
                cum_reward += reward

                done = terminated or truncated
                obs = next_obs
            env.close()
            print(f'cumulative reward: {cum_reward}')

    def save(self,filepath):
        torch.save(self.Q_fun.state_dict(),os.path.join(filepath,'final_agent.pth'))

    def load(self,filepath):
        self.Q_fun.load_state_dict(torch.load(os.path.join(filepath, 'final_agent.pth')))