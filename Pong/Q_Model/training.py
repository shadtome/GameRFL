import Q_Model.training as tr
import Space_Invaders.Q_Model.agent as agent
import gymnasium as gym

class Space_Invaders_training(tr.training):
    def add_wrappers(self):
        self.env = gym.wrappers.NormalizeReward(self.env)
        self.env = agent.Space_Invaders_Rewards(self.env)
        super().add_wrappers()
        
        

        