import Q_Model.training as tr
from gymnasium import Wrapper

class Assault_training(tr):
    def add_wrapper(self):
        super().add_wrapper()
        self.env = Assault_Rewards(self.env)


class Assault_Rewards(Wrapper):
    def __init__(self,env):
        super().__init__(env)

    def step(self,action):

        new_obs, reward, terminated, truncated, info = self.env.step(action)

        # modify the reward based on the obs
        