import random
from collections import defaultdict
import numpy as np
from tic_env import TictactoeEnv

class QState:
    def __init__(self, grid: np.ndarray, action: int):
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

    def set_player(self, player = 'X', j=-1):
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def randomMove(self, grid):
        """ Chose a random move from the available options. """
        avail = self.empty(grid)
        return avail[random.randint(0, len(avail)-1)]

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

    def simulate_action(self, grid, action):
        '''simulate action to determine reward and next state'''
        env = TictactoeEnv()
        env.grid = grid
        env.current_player = self.player
        next_grid, end, winner = env.step(action)
        
        if end:
            if winner == self.player:
                return next_grid, 1
            return next_grid, -1
        
        return next_grid, 0

    def act(self, grid, **kwargs):
        # whether move in random or not
        if random.random() < self.epsilon:
            return self.randomMove(grid)

        qstate = self.greedy(grid)
        next_grid, reward = self.simulate_action(grid, qstate.action)
        next_qstate = self.greedy(next_grid)

        self.qvalues[qstate] += self.alpha * (reward + self.gamma * self.qvalues[next_qstate] - self.qvalues[qstate])

        return qstate.action