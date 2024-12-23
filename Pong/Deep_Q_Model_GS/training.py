import torch
import torch.nn as nn
import gymnasium as gym
import Pong.Deep_Q_Model_GS.agent as agent
import Deep_Q_Learning_GrayScale.training as training
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt

class Pong_training(training.Q_training):
    def __init__(self,name,start_epsilon=1, final_epsilon=0.1, n_episodes=100, update_factor=500):
        filepath = f'/home/cody/Documents/DataSciBC/GameRFL/Pong/saved_models/{name}'
        super().__init__(agent_cl=agent.Pong_Agent,filepath=filepath,start_epsilon=start_epsilon,
                         final_epsilon=final_epsilon,n_episodes=n_episodes,update_factor=update_factor)
        
        
        

