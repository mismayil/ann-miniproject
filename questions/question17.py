import sys
sys.path.append("..")

from utils import play
from deepqlearner import DeepQPlayer, ReplayMemory, DeepQNetwork

EPS_MIN = 0.1
EPS_MAX = 0.8
n_stars = [1, 50, 100, 500, 1000, 5000, 10000, 20000, 40000]

for n_star in n_stars:
    memory = ReplayMemory(10000)
    policy_net = DeepQNetwork()
    target_net = DeepQNetwork()
    q_player1 = DeepQPlayer(epsilon=lambda n, n_star=n_star: max(EPS_MIN, EPS_MAX * (1 - n / n_star)), target_update=500, batch_size=64, log_every=250,
                            policy_net=policy_net, target_net=target_net, memory=memory, wandb_name=f"q17_nstar{n_star}")
    q_player2 = DeepQPlayer(epsilon=lambda n, n_star=n_star: max(EPS_MIN, EPS_MAX * (1 - n / n_star)), target_update=500, batch_size=64, log=False,
                            policy_net=policy_net, target_net=target_net, memory=memory, swap_state=True)
    play(q_player1, q_player2, episodes=20000)