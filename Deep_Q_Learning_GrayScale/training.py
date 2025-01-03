import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import os
import device



class Q_training:
    def __init__(self,agent_cl,filepath,start_epsilon=1, final_epsilon=0.1, n_episodes=100, update_factor=500):
        
        # The agent with the Q_function
        self.agent = agent_cl()

        # The target Q_net
        self.target_Q_net = self.agent.Q_fun.copy()
        self.target_Q_net.to(device.Device)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # file path for saving and loading
        self.filepath = filepath
        if os.path.exists(self.filepath)==False:
            os.mkdir(self.filepath)

        # WHere to start and end the number of episodes
        self.start_episodes = 0
        self.n_episodes = n_episodes

        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon

        # Learning Rate
        self.lr = 1e-3

        # Used for training
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.agent.Q_fun.parameters(),lr=self.lr,weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer,step_size=100,gamma=0.5)

        # results from training
        self.losses = []
        self.cum_reward = []
        self.update_factor = update_factor
        self.actions_taken = []


    def soft_update(self,tau):
        for target_param, policy_param in zip(self.target_Q_net.parameters(),self.agent.Q_fun.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def sample_replay(self, batch_size):

        sample = random.choices(self.replay_buffer,k=batch_size)
        state, action, next_state, reward,terminated = zip(*sample)

        state_tensor = torch.tensor(np.array(state),dtype=torch.float32,device=device.DEVICE)
        #state_tensor = state_tensor.permute((0,3,1,2))
        #state_tensor = state_tensor.unsqueeze(1)

        action_tensor = torch.tensor(np.array(action), dtype=torch.int64)

        next_state_tensor = torch.tensor(np.array(next_state),dtype=torch.float32,device=device.DEVICE)
        #next_state_tensor = next_state_tensor.permute((0,3,1,2))
        #next_state_tensor = next_state_tensor.unsqueeze(1)

        reward_tensor = torch.tensor(np.array(reward),dtype=torch.float32,device=device.DEVICE)
        terminated_tensor = torch.tensor(np.array(terminated),dtype = torch.int64,device=device.DEVICE)
        return state_tensor,action_tensor,next_state_tensor,reward_tensor, terminated_tensor

    def compute_loss(self,batch_size, gamma=0.97):
        state, actions,next_states, rewards, terminated = self.sample_replay(batch_size)
        
        q_values = self.agent.Q_fun(state).gather(1,actions.unsqueeze(1)).squeeze(1)
        next_action = self.agent.Q_fun(next_states).argmax(dim=1)
        next_q_values = self.target_Q_net(next_states).gather(1, next_action.unsqueeze(1)).squeeze(1)
        #next_q_values = self.target_Q_net(next_states).max(1)[0]
        target = rewards + gamma*next_q_values*(1-terminated)
        return self.loss_fun(q_values,target)
    
    def decay_epsilon(self,episode,n_episodes):
        start_epsilon = self.start_epsilon
        final_epsilon = self.final_epsilon
       
        self.epsilon = max(final_epsilon,start_epsilon + ((final_epsilon-start_epsilon)/n_episodes)*episode )
        print(self.epsilon)

    def train(self,reward_wrapper):
        n_episodes = self.n_episodes
        env = gym.make(self.agent.env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_episodes)
        
        env = self.agent.add_wrappers(env)
        env = gym.wrappers.NormalizeObservation(env)
        


        update_target = 0
        for ep in range(self.start_episodes,n_episodes):
            # First, we need to get some experience for the agent
            state,info = env.reset()
            print(f'episode {ep}')
            done = False
            
            cum_reward = 0

            while not done:
                
                action = self.agent.get_action(env,state,self.epsilon)
                self.actions_taken.append(action)
                next_state, reward, terminated, truncated, info = env.step(action)
                self.replay_buffer.append([state,action,next_state,reward,terminated])
                
                cum_reward += reward

                state = next_state

                done = truncated or terminated
                
                if len(self.replay_buffer)>1000:
                    #self.target_Q_net.load_state_dict(self.agent.Q_fun.state_dict())
                    self.soft_update(tau=0.001)

                update_target+=1
                
                if len(self.replay_buffer)>1000:
                    
                    self.optimizer.zero_grad()
                    loss = self.compute_loss(batch_size=64)
                    self.losses.append(loss.detach().to('cpu').numpy())
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    if update_target % 128 ==0:
                        print(f'episode {ep} with loss {loss}')



            self.cum_reward.append(cum_reward)
            print(f'cumulative reward: {cum_reward}')
            self.decay_epsilon(ep,n_episodes)
        self.dist_rewards(env,reward_wrapper)
        env.close()
                
    def results(self):
        
        
        fig, axe = plt.subplots(2,2,figsize=(10,10))
        axe[0][0].plot(self.losses)
        axe[0][0].set_xlabel('episode')
        axe[0][0].set_ylabel('loss')
        axe[0][0].set_title('losses')

        axe[0][1].plot(self.cum_reward)
        axe[0][1].set_xlabel('episode')
        axe[0][1].set_ylabel('cum awards')
        axe[0][1].set_title('cumulative awards across episodes')

        axe[1][0].hist(self.actions_taken)
        axe[1][0].set_title('histogram of actions')
        plt.savefig(os.path.join(self.filepath,'results.png'))
        


    def dist_rewards(self,env,reward_wrapper):
        # First check if the there is an enviroment with the specified reward wrapper
        wrap = get_wrapper(env,reward_wrapper)
        if wrap!=None:
            plt.figure(figsize=(10,10))
            keys = list(wrap.dist_rewards.keys())
            values = list(wrap.dist_rewards.values())
            plt.bar(keys,values)
            plt.savefig(os.path.join(self.filepath,'rewards_dist.png'))

    def save_checkpoint(self, episode):
        if os.path.exists(self.filepath)==False:
            os.mkdir(self.filepath)

        training_checkpoint = {
            'agent' : self.agent.Q_fun.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
            'n_episodes' : self.n_episodes,
            'scheduler' : self.scheduler.state_dict(),
            'losses': self.losses,
            'cum_rewards': self.cum_reward,
            'actions_taken' : self.actions_taken,
            'replay_buffer': self.replay_buffer,
            'epsilon' : self.epsilon,
            'final_epsilon': self.final_epsilon,
        }
        torch.save(training_checkpoint, os.path.join(self.filepath, 'training.pth'))

    def save(self):

        torch.save(self.agent.Q_fun.state_dict(),os.path.join(self.filepath,'final_agent.pth'))
        

    def load(self):

        
        checkpoint = torch.load(os.path.join(self.filepath,'training.pth'))
        

        self.agent.Q_fun.load_state_dict(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_episodes = checkpoint['episode']
        self.n_episodes = checkpoint['n_episodes']
        self.losses = checkpoint['losses']
        self.cum_reward = checkpoint['cum_rewards']
        self.actions_taken = checkpoint['dis']
        self.replay_buffer = checkpoint['replay_buffer']
        


    
def get_wrapper(env,wrapper_type):
    """Retrieve a specific wrapper from a Gymnasium environment."""
    current_env = env
    while isinstance(current_env, gym.Wrapper):
        if isinstance(current_env, wrapper_type):
            return current_env
        current_env = current_env.env  # Move to the next layer
    return None  # Wrapper not found
