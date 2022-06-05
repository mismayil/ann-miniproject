import sys
sys.path.append("..")
from utils import play
from qlearner import QPlayer
from collections import defaultdict

EPS_MIN = 0.1
EPS_MAX = 0.8
n_stars = [1, 50, 100, 1000, 5000, 10000, 20000, 40000]

for n_star in n_stars:
    get_epsilon = lambda n, n_star=n_star: max(EPS_MIN, EPS_MAX * (1 - n / n_star))
    qvalues = defaultdict(int)
    q_player1 = QPlayer(epsilon=get_epsilon, qvalues=qvalues, log_every=250, test_every=250, wandb_name=f"q8_nstar{n_star}")
    q_player2 = QPlayer(epsilon=get_epsilon, qvalues=qvalues, log_every=None)
    play(q_player1, q_player2, episodes=20000)