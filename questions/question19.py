import sys
sys.path.append("..")

from utils import play
from deepqlearner import DeepQPlayer, ReplayMemory, DeepQNetwork

EPS_MIN = 0.1
EPS_MAX = 0.8
N_STAR = 1000

memory = ReplayMemory()
policy_net = DeepQNetwork()
target_net = DeepQNetwork()
epsilon = lambda n: max(EPS_MIN, EPS_MAX * (1 - n / N_STAR))
q_player1 = DeepQPlayer(epsilon=epsilon, policy_net=policy_net, target_net=target_net, memory=memory, log=True, wandb_name=f"q19_nstar{N_STAR}")
q_player2 = DeepQPlayer(epsilon=epsilon, policy_net=policy_net, target_net=target_net, memory=memory, log=False, do_optimize=False)
play(q_player1, q_player2, episodes=20000)
q_player1.save_pretrained("q19_model")