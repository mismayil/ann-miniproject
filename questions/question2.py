import sys
sys.path.append("..")
from utils import play
from qlearner import QPlayer
from tic_env import OptimalPlayer

EPS_MIN = 0.1
EPS_MAX = 0.8
n_stars = [1, 100, 1000, 10000, 40000]

for n_star in n_stars:
    get_epsilon = lambda n, n_star=n_star: max(EPS_MIN, EPS_MAX * (1 - n / n_star))
    suboptimal_player = OptimalPlayer(epsilon=0.5)
    q_player = QPlayer(epsilon=get_epsilon, log_every=250, wandb_name=f"q2_nstar{n_star}")
    play(suboptimal_player, q_player, episodes=20000)