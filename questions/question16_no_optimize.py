import sys
sys.path.append("..")

from utils import play
from deepqlearner import DeepQPlayer, ReplayMemory, DeepQNetwork

epsilons = [0.001, 0.01, 0.1, 0.2]

for eps in epsilons:
    memory = ReplayMemory(10000)
    policy_net = DeepQNetwork()
    target_net = DeepQNetwork()
    q_player1 = DeepQPlayer(epsilon=eps, target_update=500, batch_size=64, log_every=250,
                            policy_net=policy_net, target_net=target_net, memory=memory, wandb_name=f"q16_no_optimize_eps{eps}")
    q_player2 = DeepQPlayer(epsilon=eps, target_update=500, batch_size=64, log=False,
                            policy_net=policy_net, target_net=target_net, memory=memory, swap_state=True, do_optimize=False)
    play(q_player1, q_player2, episodes=20000)