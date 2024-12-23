import gymnasium as gym
from tqdm import tqdm
import Q_Model.agent as agent
import matplotlib.pyplot as plt
import numpy as np
import ale_py
from collections import deque
import importlib
importlib.reload(agent)
gym.register_envs(ale_py)

class training:
    def __init__(self,env_name,learning_rate,n_episodes,start_epsilon,final_epsilon,discount_factor=0.95):
        
        self.lr = learning_rate
        self.n_episodes = n_episodes
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = start_epsilon/(n_episodes/2)


        # Create the enviroment for training
        self.env_name = env_name
        self.env = gym.make(env_name,obs_type='ram')
        self.add_wrappers()

        self.agent = agent.Agent(self.env.action_space.n,learning_rate=learning_rate,initial_epsilon=start_epsilon,
                                        epsilon_decay=start_epsilon/(n_episodes/2),final_epsilon=final_epsilon,
                                        discount_factor=discount_factor)
        
    def add_wrappers(self):
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env,buffer_length=self.n_episodes)
        
        

    def train(self):

        
        # Make the training env be the main enviroment we evaluate
        for ep in tqdm(range(self.n_episodes)):
            obs, info = self.env.reset()
            done = False

            obs = tuple(map(tuple, np.vstack(obs.T)))
            episode_reward = 0

            # play one episode
            while not done:
                action = self.agent.get_action(obs,self.env)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_obs = tuple(map(tuple, np.vstack(next_obs.T)))
                self.agent.update(obs, action,reward,terminated,next_obs)
                done = terminated or truncated
                obs = next_obs

                episode_reward += reward
            
            #print(f'episode {ep} with reward {episode_reward}')

            
            self.agent.decay_epsilon()
            
        self.env.close()     

    def training_metrics(self):
        # visualize the episode rewards, episode length and training error in one figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        

        # np.convolve will compute the rolling mean for the number of episodes
        axs[0][0].plot(np.convolve(self.env.return_queue, np.ones(self.n_episodes)))
        axs[0][0].set_title("Episode Rewards")
        axs[0][0].set_xlabel("Episode")
        axs[0][0].set_ylabel("Reward")

        axs[0][1].plot(np.convolve(self.env.length_queue, np.ones(self.n_episodes)))
        axs[0][1].set_title("Episode Lengths")
        axs[0][1].set_xlabel("Episode")
        axs[0][1].set_ylabel("Length")

        axs[0][2].plot(np.convolve(self.agent.training_error, np.ones(self.n_episodes)))
        axs[0][2].set_title("Training Error")
        axs[0][2].set_xlabel("Episode")
        axs[0][2].set_ylabel("Temporal Difference")

        plt.tight_layout()
        plt.show()

