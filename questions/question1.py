import sys
sys.path.append("..")
from utils import play
from qlearner import QPlayer
from tic_env import OptimalPlayer

epsilons = [0.01, 0.1, 0.2, 0.5, 0.75]

for eps in epsilons:
    suboptimal_player = OptimalPlayer(epsilon=0.5)
    q_player = QPlayer(epsilon=eps, log_every=250, wandb_name=f"q1_eps{eps}")
    play(suboptimal_player, q_player, episodes=20000)
    