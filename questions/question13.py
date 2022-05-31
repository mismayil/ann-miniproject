import sys
sys.path.append("..")

from utils import play
from deepqlearner import DeepQPlayer
from tic_env import OptimalPlayer

EPS_MIN = 0.1
EPS_MAX = 0.8
n_stars = [1, 50, 100, 500, 1000, 5000, 10000, 20000, 40000]

for n_star in n_stars:
    optimal_player = OptimalPlayer(epsilon=0.5)
    q_player = DeepQPlayer(epsilon=lambda n, n_star=n_star: max(EPS_MIN, EPS_MAX * (1 - n / n_star)), target_update=500, log_every=250, wandb_name=f"q13_nstar{n_star}")
    play(optimal_player, q_player, episodes=20000)