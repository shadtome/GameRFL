
import gymnasium as gym
import ale_py
import numpy as np
from collections import defaultdict
from gymnasium.spaces import Box
from gymnasium.utils import play
import time

gym.register_envs(ale_py)

#env = gym.make('ALE/Assault-v5',render_mode='human')
env = gym.make('ALE/SpaceInvaders-v5',render_mode='human',continuous=False)
observation, info = env.reset()
q_values = defaultdict(lambda: np.zeros(env.action_space.n))
#ale = env.unwrapped.ale
print(info)
print(env.observation_space)
print(env.action_space)
#print(env.observation_space.sample())
print(env.unwrapped.get_action_meanings())

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(2)
    ale = env.unwrapped.ale
    print(f'missiles_y: {ale.getRAM()[9]}')
    print(f'enemies_x: {ale.getRAM()[28]}')
    print(f'enemies_y: {ale.getRAM()[24]}')
    
    

    episode_over = terminated or truncated

env.close()
