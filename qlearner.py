from typing import Union, Tuple
import random
from collections import defaultdict
import numpy as np
from tic_env import TictactoeEnv
from utils import print_qstate

class QState:
    def __init__(self, grid: np.ndarray, action: Union[int, Tuple[int, int]]):
        self.grid = grid
        self.state = tuple(grid.ravel().tolist())
        self.action = action

    def __hash__(self) -> int:
        return hash((self.state, self.action))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, QState) and self.state == other.state and self.action == other.action

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

    def set_player(self, player = 'X', j=-1):
        self.player = player
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

    def simulate(self, grid, action):
        '''simulate action to determine reward and next state'''
        env = TictactoeEnv()
        env.grid = grid
        env.current_player = self.player
        next_grid, end, winner = env.step(action)
        reward = 0

        if end:
            if winner == self.player:
                reward = 1
            else:
                reward = -1
        
        return next_grid, end, reward

    def act(self, grid, **kwargs):
        next_qstate = self.greedy(grid)

        if random.random() < self.epsilon:
            print("Playing random")
            qstate = self.random(grid)
        else:
            print("Playing greedy")
            qstate = next_qstate

        print("Current qstate:")
        print_qstate(qstate)

        _, end, reward = self.simulate(grid, qstate.action)
        print(f"Simulation: {end=}, {reward=}")

        if end:
            self.qvalues[qstate] += self.alpha * (reward - self.qvalues[qstate])
            self.last_qstate = None
        else:
            if self.last_qstate:
                self.qvalues[self.last_qstate] += self.alpha * (reward + self.gamma * self.qvalues[next_qstate] - self.qvalues[self.last_qstate])
            self.last_qstate = qstate

        print("Current qvalues:")
        print(self.qvalues)
        print("\n")
        return qstate.action