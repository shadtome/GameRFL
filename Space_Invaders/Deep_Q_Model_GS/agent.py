import torch
import gymnasium as gym
from gymnasium import wrappers
import ale_py
from torch import nn
from collections import deque
import numpy as np
import Space_Invaders.wrappers.rewards as rewards
import Deep_Q_Learning_GrayScale.agent as agent



# Here we are going to create a an agent designed for deep $Q$-neural networks.


class SI_Q_fun(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()

        # input sizes for the Q-function, i.e. $Q(s,a)$ where $s$ is the state, the image, and $a$ is the actions
        self.state_shape = state_shape
        self.n_actions = n_actions

        # The neural network
        """
        self.Q_function = nn.Sequential(
            nn.Conv2d(state_shape[2],16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Flatten(),
            nn.Linear(16*state_shape[0]*state_shape[1],32),
            nn.ReLU(),
            nn.Linear(32,n_actions)
        )
        """
        
        self.Q_function = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),  # Increase filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Add a second conv layer
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (state_shape[0] // 4) * (state_shape[1] // 4), 128),  # Adjust dimensions
            nn.ReLU(),
            nn.Linear(128, n_actions)
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
        copy = SI_Q_fun(self.state_shape,self.n_actions)
        copy.load_state_dict(self.state_dict())
        return copy
    

class SI_Agent(agent.Agent):
    def __init__(self):
        env_name = 'ALE/SpaceInvaders-v5'
        super().__init__(env_name,SI_Q_fun)

    def add_wrappers(self,env):
        #env = rewards.Space_Invaders_Rewards(env)
        env = wrappers.NormalizeReward(env)
        env = super().add_wrappers(env)
        return env
    
    def save(self,name):
        fd = f'/home/cody/Documents/DataSciBC/GameRFL/Space_Invaders/saved_models/{name}'
        super().save(fd)
    
    def load(self,name):
        fd = f'/home/cody/Documents/DataSciBC/GameRFL/Space_Invaders/saved_models/{name}'
        super().load(fd)

