
from gymnasium import Wrapper
from Space_Invaders.game_info import RAM_info
import ale_py
from collections import deque

class Space_Invaders_Rewards(Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.game_data = {'invaders_left_count': 0, 'player_score' : 0, 'num_lives': 0, 
                          'player_x': [None,None,None,None,None],'missles_y': 0}
        self.time_counter = 0
        self.death_mult = 0.001
        #self.mult_increment = 0.01
        self.mult_increment = 0
        self.visted  = deque(maxlen=5)
        self.actions = deque(maxlen=3)
        self.dist_rewards = {'score': 0, 'death': 0, 'new actions':0, 'new places': 0, 'staying alive': 0,
                             'misses': 0, 'rep_actions': 0}
        

    def step(self,action):

        # awards distribution
        death_reward = -500
        #new_action_reward = 10
        #new_places_reward = 10
        staying_alive_award = 0.5

        new_obs, reward, terminated, truncated, info = self.env.step(action)
        ale = self.env.unwrapped.ale
        new_reward = 0
        
        
        # Dynamic exploration rewards
        if action not in self.actions:
            new_action_reward = 100
            #new_action_reward = max(100 - len(self.actions), 1)  # Diminishing reward for new actions
            self.dist_rewards['new actions'] += new_action_reward
            new_reward += new_action_reward
            self.actions.append(action)

        # reward for seeing new places
        if ale.getRAM()[RAM_info['player_x']] not in self.visted:
            new_places_reward = 100
            #new_places_reward = max(100 - len(self.visted), 1)  # Diminishing reward for new places
            self.dist_rewards['new places'] += new_places_reward
            new_reward += new_places_reward
            self.visted.append(ale.getRAM()[RAM_info['player_x']])

        # Reward destroying invaders
        state_n_invaders = ale.getRAM()[RAM_info['invaders_left_count']]
        if  state_n_invaders< self.game_data['invaders_left_count']:
            invader_reward = 50  # Reward for destroying an invader
            self.dist_rewards['score'] += invader_reward*reward + invader_reward
            new_reward += invader_reward 
            self.game_data['invaders_left_count'] = state_n_invaders
            self.dist_rewards['score']+=invader_reward

        # Penalize missing shots
        if ale.getRAM()[RAM_info['missles_y']] != 30:  # Indicates a missed shot
            miss_penalty = -20
            self.dist_rewards['misses'] +=miss_penalty
            new_reward += miss_penalty

        # Penatly for death
        state_lives = int(ale.getRAM()[RAM_info['num_lives']])
        if state_lives < self.game_data['num_lives']:
            
            self.dist_rewards['death']+=death_reward * (4 - state_lives)
            new_reward += death_reward * (4 - state_lives)
            self.game_data['num_lives'] = state_lives
        

        
        # reward for new places visited
        if ale.getRAM()[RAM_info['player_x']] not in self.visted:
            
            self.dist_rewards['new places']+=new_places_reward
            new_reward += new_places_reward
            self.visted.append(ale.getRAM()[RAM_info['player_x']])


        # Penalize repetitive actions
        if len(set(self.actions)) == 1:  # All recent actions are the same
            rep_actions = -500
            self.dist_rewards['rep_actions'] += rep_actions
            new_reward += rep_actions  # Small penalty for lack of diversity
                
        # Reward survival proportionally to episode length
        staying_alive_award = 5 * (1 + self.time_counter / 1000)  # Increases as time progresses
        self.dist_rewards['staying alive'] += staying_alive_award
        new_reward += staying_alive_award
        
        

        return new_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.game_data['invaders_left_count'] = 36
        self.game_data['player_score'] = 0
        self.game_data['num_lives'] = 3
        self.game_data['player_x'] = [None,None,None,None,None]
        self.life_counter=0
        self.death_mult += self.death_mult*self.mult_increment
        return obs, info