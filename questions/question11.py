import sys
sys.path.append("..")

from utils import play
from deepqlearner import DeepQPlayer
from tic_env import OptimalPlayer

epsilons = [0.001, 0.01, 0.1, 0.2]

for eps in epsilons:
    suboptimal_player = OptimalPlayer(epsilon=0.5)
    q_player = DeepQPlayer(epsilon=eps, target_update=500, batch_size=64, log_every=250, wandb_name=f"q11_eps{eps}")
    play(suboptimal_player, q_player, episodes=20000)