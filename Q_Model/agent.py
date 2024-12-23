from collections import defaultdict
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class Agent:
    def __init__(self,action_space_n, learning_rate: float, initial_epsilon: float,epsilon_decay: float,final_epsilon:float,
                 discount_factor: float = 0.95):
        """ Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values),
        a learning rate and an epsilon.
        """

        self.q_values = defaultdict(lambda: np.zeros(action_space_n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        

    def get_action(self, obs,env,play=False) -> int:

        # with probability epsilon return a random action to explore the enviroment
        if np.random.random()<self.epsilon and play==False:
            return env.action_space.sample()
        # with probability $(1-\epsilon) act greedily ( exploit)
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(self, obs, action: int, reward: float,terminated: bool,
               next_obs):
        """Update the Q-values of an action"""

        future_q_value = (not terminated) * (np.max(self.q_values[next_obs]))

        temporal_diff = reward + self.discount_factor*future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] += self.lr*temporal_diff

        self.training_error.append(temporal_diff)
        

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,self.epsilon - self.epsilon_decay)

    def add_wrappers(self,env):
        return env

    def run_agent(self,env_name: str):
        # make enviroment for testing
        env = gym.make(env_name,render_mode='human',obs_type='ram')

        env = self.add_wrappers(env)

        obs, info = env.reset()
        obs = tuple(map(tuple,np.vstack(obs.T)))
        print(f'initial: {obs}\n')
        done = False

        cum_reward = 0
        # Play
        while not done:
            
            action = self.get_action(obs,env)
        
            next_obs, reward, terminated, truncated, info = env.step(action)
            cum_reward += reward
            next_obs = tuple(map(tuple, np.vstack(next_obs.T)))

            done = terminated or truncated
            obs = next_obs
        env.close()
        print(cum_reward)