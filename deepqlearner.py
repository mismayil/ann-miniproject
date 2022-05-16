from collections import namedtuple, deque
import torch
import random
import torch.nn as nn
import torch.optim as optim

VALUE_TO_PLAYER = {-1: 'O', 1: 'X', 0: None}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
    def __init__(self, epsilon=0.1, gamma=0.99, player='X', memory_capacity=10000, target_update=500, batch_size=64, learning_rate=5e-4) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.player = player
        self.memory = ReplayMemory(memory_capacity)
        self.policy_net = DeepQNetwork()
        self.target_net = DeepQNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.last_state = None
        self.last_action = None
        self.num_games = 0
        self.target_update = target_update
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def set_player(self, player = 'X', j=-1):
        self.player = player
        self.last_qstate = None
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

    def greedy(self, grid):
        with torch.no_grad():
            prediction = self.policy_net.predict(self.grid_to_state(grid).unsqueeze(0))
            return prediction.item()

    def decide(self, grid):
        if random.random() > self.epsilon:
            return self.greedy(grid)
        return self.random(grid)

    def act(self, grid):
        state = self.grid_to_state(grid)
        action = self.decide(grid)
        if self.last_state is not None:
            self.memory.push(self.last_state, self.last_action, state, 0)
        self.optimize()
        self.last_state = state
        self.last_action = action
        return action

    def end(self, grid, winner):
        self.num_games += 1
        reward = 0

        if winner == self.player:
            reward = 1
        elif winner == self.opponent():
            reward = -1

        self.memory.push(self.last_state, self.last_action, None, reward)
        self.optimize()
        self.last_state = None
        self.last_action = None

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]
        if non_final_next_states:
            non_final_next_states = torch.stack(non_final_next_states)
        else:
            non_final_next_states = None

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action).view(-1, 1)
        reward_batch = torch.stack(batch.reward).view(-1, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((self.batch_size, 1))
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].view(-1, 1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.num_games % self.target_update == 0:
            print(f"num_games={self.num_games}, loss={loss.item()}")
            self.target_net.load_state_dict(self.policy_net.state_dict())


