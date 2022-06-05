import sys
sys.path.append("..")
from utils import play
from qlearner import QPlayer
from tic_env import OptimalPlayer

N_STAR = 100
EPS_MIN = 0.1
EPS_MAX = 0.8
get_epsilon = lambda n: max(EPS_MIN, EPS_MAX * (1 - n / N_STAR))

epsilons = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

for eps in epsilons:
    other_player = OptimalPlayer(epsilon=eps)
    q_player = QPlayer(epsilon=get_epsilon, log_every=250, test_every=250, wandb_name=f"q4_eps{eps}")
    play(other_player, q_player, episodes=20000)