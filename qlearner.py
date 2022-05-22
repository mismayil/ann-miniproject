from typing import Union, Tuple
import random
from collections import defaultdict
import numpy as np
from utils import calculate_m_opt, calculate_m_rand


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


class QPlayer:
    def __init__(self, epsilon=0.01, alpha=0.05, gamma=0.99, player='X', log_every=250, test_every=None, *args, **kwargs):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.player = player # 'X' or 'O'
        self.qvalues = defaultdict(int)
        self.last_qstate = None
        self.last_reward = 0
        self.num_games = 0
        self.running_reward = 0
        self.running_win = 0
        self.running_loss = 0
        self.avg_rewards = []
        self.avg_wins = []
        self.avg_losses = []
        self.m_values = {"m_opt": [], "m_rand": []}
        self.log_every = log_every
        self.log = False if log_every is None or log_every <=0 else True
        self.test_every = test_every
        self.test = False if test_every is None or log_every <=0 else True
        self.eval_mode = False
        
        
    def set_player(self, player = 'X', j=-1):
        self.player = player
        self.last_qstate = None
        self.last_reward = 0
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'
            
    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def random(self, grid):
        """ Chose a random action from the available options. """
        avail = self.empty(grid)
        return QStateAction(grid, avail[random.randint(0, len(avail)-1)])

    def empty(self, grid):
        '''return all empty positions'''
        avail = np.where(grid==0)
        return list(zip(*avail))

    def greedy(self, grid):
        '''choose next state greedily'''
        actions = self.empty(grid)
        best_qstates = None
        max_qvalue = None

        for action in actions:
            qstate = QStateAction(grid, action)
            qvalue = self.qvalues.get(qstate, 0)
            
            if max_qvalue is None or qvalue > max_qvalue:
                max_qvalue = qvalue
                best_qstates = [qstate]
            elif max_qvalue is not None and qvalue==max_qvalue:
                best_qstates.append(qstate)
        
        return np.random.choice(best_qstates)

    def opponent(self):
        return 'X' if self.player == 'O' else 'O'

    def decide(self, grid):
        epsilon = self.epsilon(self.num_games) if callable(self.epsilon) else self.epsilon
        if self.eval_mode or random.random() < epsilon:
            return self.random(grid)
        return self.greedy(grid)

    def update(self, grid, reward=0, end=False):
        next_value = 0

        if not end:
            next_value = self.qvalues[self.greedy(grid)]

        if self.last_qstate:
            self.qvalues[self.last_qstate] += self.alpha * (reward + self.gamma * next_value - self.qvalues[self.last_qstate])

    def end(self, grid, winner):
        if self.eval_mode: return
        
        self.num_games += 1
        reward, win, loss = 0, 0, 0

        if winner == self.player:
            reward, win, loss = 1, 1, 0
        elif winner == self.opponent():
            reward, win, loss = -1, 0, 1

        self.update(grid, reward=reward, end=True)

        self.last_qstate = None
        
        if self.log:
            self.running_win += win
            self.running_loss += loss
            self.running_reward += reward
            
            if (self.num_games+1) % self.log_every == 0:
                self.avg_wins.append(self.running_win / self.log_every)
                self.running_win = 0
                
                self.avg_losses.append(self.running_loss / self.log_every)
                self.running_loss = 0
                
                self.avg_rewards.append(self.running_reward / self.log_every)
                self.running_reward = 0
                
        if self.test:
            if  (self.num_games+1) % self.test_every == 0:
                m_opt = calculate_m_opt(self)
                m_rand = calculate_m_rand(self)

                self.m_values["m_opt"].append(m_opt)
                self.m_values["m_rand"].append(m_rand)
                

    def act(self, grid):
        qstate = self.decide(grid)
        self.update(grid)
        self.last_qstate = qstate
        return qstate.action