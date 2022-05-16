from typing import Union, Tuple
import random
from collections import defaultdict
import numpy as np

POLICY_EPS_GREEDY = "eps-greedy"
METHOD_Q = "method-q"
METHOD_SARSA = "method-sarsa"
METHOD_EXP_SARSA = "method-exp-sarsa"

class QStateAction:
    def __init__(self, grid: np.ndarray, action: Union[int, Tuple[int, int]]):
        self.grid = grid
        self.state = tuple(grid.ravel().tolist())
        self.action = action

    def __hash__(self) -> int:
        return hash((self.state, self.action))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, QStateAction) and (self.state == other.state) and self.action == other.action

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def __repr__(self) -> str:
        return f"state={self.state}, action={self.action}"


class TDPlayer:
    def __init__(self, epsilon=0.2, alpha=0.05, gamma=0.99, player='X', policy=POLICY_EPS_GREEDY, method=METHOD_Q):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.player = player # 'X' or 'O'
        self.qvalues = defaultdict(lambda:np.zeros((3,3)))
        self.last_qstate = None
        self.last_reward = 0
        self.policy = policy
        self.method = method

    def set_player(self, player = 'X', j=-1):
        self.player = player
        self.last_qstate = None
        self.last_reward = 0
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def random(self, grid):
        """ Chose a random action from the available options. """
        avail = self.empty(grid)
        random_int = random.randint(0, len(avail[0])-1)
        random_action = (avail[0][random_int], avail[1][random_int])
        return QStateAction(grid, random_action)

    def empty(self, grid):
        '''return all empty positions'''
        return np.where(grid==0)

    def greedy(self, grid):
        '''choose next state greedily'''
        actions = self.empty(grid)
        best_qstate = None

        state = tuple(grid.ravel().tolist())
        qvalues = self.qvalues.get(state, np.zeros((3,3)))
        qvalues = qvalues[actions]
        
        max_val = np.max(qvalues)

        best_action_id = np.random.choice(np.flatnonzero(qvalues == max_val))
        best_action = (actions[0][best_action_id], actions[1][best_action_id])
        best_qstate = QStateAction(grid, best_action)
        
        # print('grid', grid)
        # print('state', state)
        # print('actions', actions)
        # print('qvalues', qvalues)
        # print('max_val', max_val)
        # print('flat', np.flatnonzero(qvalues == max_val))
        # print('action_id', best_action_id)
        # print('best_action', best_action)        
        
        return best_qstate

    def opponent(self):
        return 'X' if self.player == 'O' else 'O'

    def decide(self, grid):
        if self.policy == POLICY_EPS_GREEDY:
            if random.random() < self.epsilon:
                return self.random(grid)
            else:
                return self.greedy(grid)

        return self.random(grid)

    def update(self, grid, reward=0, end=False):
        next_value = 0

        if not end:
            if self.method == METHOD_Q:
                next_state = self.greedy(grid)
                next_value = self.qvalues[next_state.state][next_state.action]

        if self.last_qstate:
            self.qvalues[self.last_qstate.state][self.last_qstate.action] += self.alpha * (reward + self.gamma * next_value - self.qvalues[self.last_qstate.state][self.last_qstate.action])

    def end(self, grid, winner):
        reward = 0

        if winner == self.player:
            reward = 1
        elif winner == self.opponent():
            reward = -1

        self.update(grid, reward=reward, end=True)

        self.last_qstate = None

    def act(self, grid):
        qstate = self.decide(grid)
        self.update(grid)
        self.last_qstate = qstate
        return qstate.action