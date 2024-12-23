import pathlib
import sys
parentdir = pathlib.Path.cwd()
sys.path.append(str(parentdir))
print(parentdir)
import Pong.Deep_Q_Model.training as trainer




SI_train = trainer.Pong_training(name='test_2',start_epsilon=1,n_episodes=5,final_epsilon=0.1,update_factor=100)
SI_train.train()

SI_train.save()
SI_train.results()