import sys
sys.path.append("..")
from utils import play
from qlearner import QPlayer
from collections import defaultdict

epsilons = [0.01, 0.1, 0.2, 0.5]

for eps in epsilons:
    qvalues = defaultdict(int)
    q_player1 = QPlayer(epsilon=eps, qvalues=qvalues, log_every=250, test_every=250, wandb_name=f"q7_eps{eps}")
    q_player2 = QPlayer(epsilon=eps, qvalues=qvalues, log_every=None)
    play(q_player1, q_player2, episodes=20000)