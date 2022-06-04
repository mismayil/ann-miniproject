from collections import namedtuple, deque
import torch
import random
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from utils import calculate_m_opt, calculate_m_rand

VALUE_TO_PLAYER = {-1: 'O', 1: 'X', 0: None}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state=None, reward=0):
        """Save a transition"""
        self.memory.append(Transition(state=state.clone(),
                                      action=action.clone() if isinstance(action, torch.Tensor) else torch.tensor(action),
                                      next_state=next_state.clone() if next_state is not None else None,
                                      reward=reward.clone() if isinstance(reward, torch.Tensor) else torch.tensor(reward)))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):
    def __init__(self, in_dim=18, out_dim=9, hidden_dim=128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.model(x.flatten(start_dim=1))
    
    def predict(self, x):
        predictions = self.forward(x)
        return torch.argmax(predictions)


class DeepQPlayer:
    def __init__(self, epsilon=0.1, gamma=0.99, player='X', memory_capacity=10000, target_update=500,
                 batch_size=64, learning_rate=5e-4, log_every=250, debug=False,
                 policy_net=None, target_net=None, memory=None, swap_state=False, log=True, wandb_name=None, do_optimize=True, *args, **kwargs) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.player = player
        self.memory = ReplayMemory(memory_capacity) if memory is None else memory
        self.policy_net = DeepQNetwork().to(DEVICE) if policy_net is None else policy_net.to(DEVICE)
        self.target_net = DeepQNetwork().to(DEVICE) if target_net is None else target_net.to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.num_games = 0
        self.target_update = target_update
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.running_reward = 0
        self.running_loss = 0
        self.avg_rewards = []
        self.avg_losses = []
        self.log_every = log_every
        self.debug = debug
        self.m_values = {"m_opt": [], "m_rand": []}
        self.eval_mode = False
        self.swap_state = swap_state
        self.log = log
        self.wandb_name = wandb_name
        self.wandb_run = None
        self.do_optimize = do_optimize

        if self.log and self.wandb_name is not None:
            import wandb
            self.wandb_run = wandb.init(project="ann-project", name=wandb_name, reinit=True,
                                        config={"epsilon": epsilon, "gamma": gamma, "player": player, "memory_capacity": memory_capacity,
                                                "target_update": target_update, "batch_size": batch_size, "learning_rate": learning_rate,
                                                "log_every": log_every, "debug": debug, "swap_state": swap_state, "log": log})

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def finish_run(self):
        if not self.eval_mode and self.wandb_run:
            self.wandb_run.finish()

    def save_pretrained(self, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        config = dict(epsilon=None if callable(self.epsilon) else self.epsilon, gamma=self.gamma, player=self.player, memory_capacity=len(self.memory),
                      target_update=self.target_update, learning_rate=self.learning_rate, batch_size=self.batch_size,
                      log_every=self.log_every, debug=self.debug, avg_losses=self.avg_losses, avg_rewards=self.avg_rewards, m_values=self.m_values)
        Path(save_path, "config.json").write_text(json.dumps(config))
        torch.save(self.policy_net.state_dict(), Path(save_path, "policy_net.pt"))
        torch.save(self.target_net.state_dict(), Path(save_path, "target_net.pt"))

    @classmethod
    def from_pretrained(cls, load_path):
        config = json.loads(Path(load_path, "config.json").read_text())
        policy_net = torch.load(Path(load_path, "policy_net.pt"))
        target_net = torch.load(Path(load_path, "target_net.pt"))
        player = cls(**config)
        player.policy_net.load_state_dict(policy_net)
        player.target_net.load_state_dict(target_net)
        player.avg_losses = config["avg_losses"]
        player.avg_rewards = config["avg_rewards"]
        player.m_values = config["m_values"]
        return player

    def set_player(self, player = 'X', j=-1):
        self.player = player
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def empty(self, grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i/3), i % 3)
            if grid[pos] == 0:
                avail.append(i)
        return avail

    def random(self, grid):
        """ Chose a random action from the available options. """
        avail = self.empty(grid)
        return avail[random.randint(0, len(avail)-1)]

    def opponent(self):
        return 'X' if self.player == 'O' else 'O'

    def grid_to_state(self, grid):
        state = torch.zeros((3, 3, 2))

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if VALUE_TO_PLAYER[grid[i, j]] == self.player:
                    state[i, j] = torch.tensor([1, 0])
                elif VALUE_TO_PLAYER[grid[i, j]] == self.opponent():
                    state[i, j] = torch.tensor([0, 1])
                else:
                    state[i, j] = torch.tensor([0, 0])
        
        return state

    def maybe_swap_state(self, state):
        if self.swap_state:
            return state.flip(dims=[2])
        return state

    def greedy(self, grid):
        with torch.no_grad():
            prediction = self.policy_net.predict(self.grid_to_state(grid).unsqueeze(0).to(DEVICE))
            return prediction.item()

    def decide(self, grid):
        epsilon = self.epsilon(self.num_games) if callable(self.epsilon) else self.epsilon
        if self.eval_mode or random.random() > epsilon:
            return self.greedy(grid)
        return self.random(grid)

    def act(self, grid):
        state = self.grid_to_state(grid)
        action = self.decide(grid)

        if not self.eval_mode:
            if self.last_state is not None:
                self.memory.push(self.maybe_swap_state(self.last_state), self.last_action, self.maybe_swap_state(state), self.last_reward)
            
            if self.do_optimize:
                self.optimize()

        self.last_state = state
        self.last_action = action
        self.last_reward = 0

        return action

    def end(self, grid, winner, invalid_move=False):
        if not self.eval_mode:
            self.num_games += 1
            reward = 0

            if winner == self.player:
                reward = 1
            elif winner == self.opponent() or invalid_move:
                reward = -1

            self.memory.push(self.maybe_swap_state(self.last_state), self.last_action, None, reward)

            if self.do_optimize:
                loss = self.optimize()

            self.last_state = None
            self.last_action = None

            if self.log:
                self.running_reward += reward

                if loss is not None:
                    self.running_loss += loss

                if (self.num_games+1) % self.log_every == 0:
                    avg_reward = self.running_reward / self.log_every
                    self.avg_rewards.append(avg_reward)
                    self.running_reward = 0

                    avg_loss = self.running_loss / self.log_every
                    self.avg_losses.append(avg_loss)
                    self.running_loss = 0

                    m_opt = calculate_m_opt(self)
                    m_rand = calculate_m_rand(self)

                    self.m_values["m_opt"].append(m_opt)
                    self.m_values["m_rand"].append(m_rand)

                    if self.wandb_name is not None:
                        import wandb
                        wandb.log({"avg_reward": avg_reward, "avg_loss": avg_loss, "m_opt": m_opt, "m_rand": m_rand})

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool).to(DEVICE)
        non_final_next_states = [s for s in batch.next_state if s is not None]

        if non_final_next_states:
            non_final_next_states = torch.stack(non_final_next_states).to(DEVICE)
        else:
            non_final_next_states = None

        state_batch = torch.stack(batch.state).to(DEVICE)
        action_batch = torch.stack(batch.action).view(-1, 1).to(DEVICE)
        reward_batch = torch.stack(batch.reward).view(-1, 1).to(DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((self.batch_size, 1)).to(DEVICE)

        if non_final_next_states is not None:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].view(-1, 1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.num_games % self.target_update == 0:
            if self.debug:
                print(f"num_games={self.num_games}, loss={loss.item()}")
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()