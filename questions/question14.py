import sys
sys.path.append("..")

from utils import play
from deepqlearner import DeepQPlayer
from tic_env import OptimalPlayer
import random

EPS_MIN = 0.1
EPS_MAX = 0.8
n_star = 100
epsilons = [0, 0.01, 0.1, 0.5, 1]

for eps in epsilons:
    other_player = OptimalPlayer(epsilon=eps)
    q_player = DeepQPlayer(epsilon=lambda n, n_star=n_star: max(EPS_MIN, EPS_MAX * (1 - n / n_star)), target_update=500, log_every=250, wandb_name=f"q14_eps{eps}")
    play(other_player, q_player, episodes=20000)