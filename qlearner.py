from typing import Union, Tuple
import random
from collections import defaultdict
import numpy as np

class QState:
    def __init__(self, grid: np.ndarray, action: Union[int, Tuple[int, int]]):
        self.grid = grid
        self.state = tuple(grid.ravel().tolist())
        self.action = action

    def __hash__(self) -> int:
        return hash((self.state, self.action))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, QState) and (self.state == other.state) and self.action == other.action

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def __repr__(self) -> str:
        return f"state={self.state}, action={self.action}"


class QPlayer:
    def __init__(self, epsilon=0.2, alpha=0.05, gamma=0.99, player='X'):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.player = player # 'X' or 'O'
        self.qvalues = defaultdict(int)
        self.last_qstate = None
        self.last_reward = 0

    def set_player(self, player = 'X', j=-1):
        self.player = player
        self.last_qstate = None
        self.last_reward = 0
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def random(self, grid):
        """ Chose a random action from the available options. """
        avail = self.empty(grid)
        return QState(grid, avail[random.randint(0, len(avail)-1)])

    def empty(self, grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i/3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def greedy(self, grid):
        '''choose next state greedily'''
        actions = self.empty(grid)
        best_qstate = None
        max_qvalue = None

        for action in actions:
            qstate = QState(grid, action)
            qvalue = self.qvalues.get(qstate, 0)
            if max_qvalue is None or qvalue > max_qvalue:
                max_qvalue = qvalue
                best_qstate = qstate
        
        return best_qstate

    def opponent(self):
        return 'X' if self.player == 'O' else 'O'

    def end(self, winner):
        reward = 0

        if winner == self.player:
            reward = 1
        elif winner == self.opponent():
            reward = -1

        self.qvalues[self.last_qstate] += self.alpha * (reward - self.qvalues[self.last_qstate])

        self.last_qstate = None

    def act(self, grid, **kwargs):
        greedy_qstate = self.greedy(grid)

        if random.random() < self.epsilon:
            qstate = self.random(grid)
        else:
            qstate = greedy_qstate

        if self.last_qstate:
            # reward is 0
            self.qvalues[self.last_qstate] += self.alpha * (self.gamma * self.qvalues[greedy_qstate] - self.qvalues[self.last_qstate])

        self.last_qstate = qstate

        return qstate.action