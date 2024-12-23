from gymnasium import Wrapper
import Q_Model.agent as agent
from Space_Invaders.game_info import RAM_info


class Spacer_Invaders_Agent(agent.Agent):
    def add_wrappers(self,env):
        env = Space_Invaders_Rewards(env)
        print('hellooooo')
        return env

class Space_Invaders_Rewards(Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.game_data = {'invaders_left_count': 0, 'player_score' : 0, 'num_lives': 0, 'player_x': [None,None,None],
                          'missles_y': 0}
        self.time_counter = 0
        self.death_mult = 1
        self.mult_increment = 0.01
        

    def step(self,action):

        new_obs, reward, terminated, truncated, info = self.env.step(action)
        new_reward = 0
        if new_obs[RAM_info['invaders_left_count']]<self.game_data['invaders_left_count']:
            new_reward += 20 * self.death_mult
            self.game_data['invaders_left_count'] = new_obs[RAM_info['invaders_left_count']]

        if new_obs[RAM_info['num_lives']] < self.game_data['num_lives']:
            new_reward -= 5 * self.death_mult
            self.game_data['num_lives'] = new_obs[RAM_info['num_lives']]
        
        if new_obs[RAM_info['invaders_left_count']]==0:
            new_reward +=100

        self.game_data['player_x'][0] = self.game_data['player_x'][1]
        self.game_data['player_x'][1] = self.game_data['player_x'][2]
        self.game_data['player_x'][2] = new_obs[RAM_info['player_x']]
        
        
        if self.game_data['player_x']!=[None,None,None] and self.game_data['player_x'][0] == self.game_data['player_x'][1] == self.game_data['player_x'][2]:
            new_reward -=0.5*self.death_mult
        
        
        self.life_counter+=1
        if self.life_counter % 50 == 0:
            new_reward +=10*self.death_mult
        
        

        return new_obs, new_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.game_data['invaders_left_count'] = 36
        self.game_data['player_score'] = 0
        self.game_data['num_lives'] = 3
        self.game_data['player_x'] = [None,None,None]
        self.life_counter=0
        self.death_mult += self.death_mult*self.mult_increment
        return obs, info
    
