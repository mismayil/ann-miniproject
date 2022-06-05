import random
from collections import defaultdict
import numpy as np
from utils import calculate_m_opt, calculate_m_rand


class QStateAction:
    """
    This is a helper class to store the state-action pair.
    """
    def __init__(self, grid, action):
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
    def __init__(self, epsilon=0.01, alpha=0.05, gamma=0.99, player='X', log_every=250, test_every=None, qvalues=None, wandb_name=None, *args, **kwargs):
        """Initialize a Q-learning player

        Args:
            epsilon (float, optional): Epsilon value. Defaults to 0.01.
            alpha (float, optional): Alpha value. Defaults to 0.05.
            gamma (float, optional): Gamma value. Defaults to 0.99.
            player (str, optional): Player symbol. Defaults to 'X'.
            log_every (int, optional): Logging frequency (i.e. for logging avg reward). Defaults to 250.
            test_every (int, optional): Testing frequency (i.e. for logging M_opt and M_rand). Defaults to None.
            qvalues (dict, optional): Q-values dict. Defaults to None.
            wandb_name (str, optional): Wandb run name. Defaults to None.
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.player = player # 'X' or 'O'
        self.qvalues = defaultdict(int) if qvalues is None else qvalues
        self.last_qstate = None
        self.last_reward = 0
        self.num_games = 0
        self.running_reward = 0
        self.avg_rewards = []
        self.m_values = {"m_opt": [], "m_rand": []}
        self.log_every = log_every
        self.log = False if log_every is None or log_every <=0 else True
        self.test_every = test_every
        self.test = False if test_every is None or log_every <=0 else True
        self.eval_mode = False
        self.wandb_name = wandb_name
        self.wandb_run = None

        if self.log and self.wandb_name is not None:
            import wandb
            self.wandb_run = wandb.init(project="ann-project", name=wandb_name, reinit=True,
                                        config={"epsilon": epsilon, "gamma": gamma, "player": player,
                                                "log_every": log_every, "test_every": test_every, "alpha": alpha})
        
        
    def set_player(self, player = 'X', j=-1):
        self.player = player
        self.last_qstate = None
        self.last_reward = 0
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'
            
    def eval(self):
        """
        Put player in evaluation mode. In this mode, player does not train
        and does not log metrics.
        """
        self.eval_mode = True

    def train(self):
        """
        Put player in training mode. In this mode, player trains and logs metrics.
        """
        self.eval_mode = False

    def finish_run(self):
        """Finish wandb run if present and if in training mode"""
        if not self.eval_mode and self.wandb_run:
            self.wandb_run.finish()

    def random(self, grid):
        """Chose a random action from the available options. """
        avail = self.empty(grid)
        return QStateAction(grid, avail[random.randint(0, len(avail)-1)])

    def empty(self, grid):
        """Return all empty cells from the grid."""
        avail = np.where(grid==0)
        return list(zip(*avail))

    def greedy(self, grid):
        """Return the best action according to the current Q-values."""
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
        """Get the opponent player symbol."""
        return 'X' if self.player == 'O' else 'O'

    def decide(self, grid):
        """Decide on the next action."""
        epsilon = self.epsilon(self.num_games) if callable(self.epsilon) else self.epsilon
        if self.eval_mode or random.random() > epsilon:
            return self.greedy(grid)
        return self.random(grid)

    def update(self, grid, reward=0, end=False):
        """Update the Q-values based on the last action."""
        next_value = 0

        if not end:
            next_value = self.qvalues[self.greedy(grid)]

        if self.last_qstate:
            self.qvalues[self.last_qstate] += self.alpha * (reward + self.gamma * next_value - self.qvalues[self.last_qstate])

    def end(self, grid, winner, *args, **kwargs):
        """End of game callback. Update the Q-values based on the last action and log metrics"""
        if self.eval_mode:
            return
        
        self.num_games += 1
        reward = 0

        if winner == self.player:
            reward = 1
        elif winner == self.opponent():
            reward = -1

        self.update(grid, reward=reward, end=True)

        self.last_qstate = None
        
        if self.log:
            self.running_reward += reward
            
            if (self.num_games+1) % self.log_every == 0:
                avg_reward = self.running_reward / self.log_every
                self.avg_rewards.append(avg_reward)
                self.running_reward = 0

                if self.wandb_name is not None:
                    import wandb
                    wandb.log({"avg_reward": avg_reward})
                
        if self.test:
            if  (self.num_games+1) % self.test_every == 0:
                m_opt = calculate_m_opt(self)
                m_rand = calculate_m_rand(self)

                self.m_values["m_opt"].append(m_opt)
                self.m_values["m_rand"].append(m_rand)

                if self.wandb_name is not None:
                    import wandb
                    wandb.log({"m_opt": m_opt, "m_rand": m_rand})
                

    def act(self, grid):
        """Act on the grid."""
        qstate = self.decide(grid)

        if not self.eval_mode: 
            self.update(grid)

        self.last_qstate = qstate
        return qstate.action