import gym
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

class ReplayBuffer():
    """
    1. define buffer
    2. define functions
        - put sample
        - return samples
        - calc current buffer size
    """
    def __init__(self):
        super().__init__()
        self.buffer = deque(maxlen=20000)

    def put(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []

        for state, action, reward, next_state, terminated, truncated in samples:

            states.append(state.cpu().numpy())
            actions.append(action.cpu().detach().numpy())
            rewards.append(reward)
            next_states.append(next_state.cpu().numpy())
            terminateds.append(terminated)
            truncateds.append(truncated)

        states      = torch.tensor(states, device = device)
        actions     = torch.tensor(actions, device=device)
        rewards     = torch.tensor(rewards, device=device)
        next_states = torch.tensor(next_states, device=device)
        terminateds = torch.tensor(terminateds, device=device)
        truncateds  = torch.tensor(truncateds, device=device)

        """
        print(type(states))  # tensor
        print(type(actions))  # tensor
        print(type(rewards))  # numpy
        print(type(next_states))  # tensor
        print(type(terminateds))  # bool
        print(type(truncateds))  # bool
        """

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module): # Q function approximation
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state, action):
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))

        concatenate = torch.cat([h1, h2], dim = -1)

        Q = F.relu(self.fc1(concatenate))
        Q = F.relu(self.fc2(Q))

        return self.out(Q)

class PNetwork(nn.Module): # Policy approximation
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)   # Input  : state
        self.fc2 = nn.Linear(64, 64)  # Input  : state
        self.fc3 = nn.Linear(64, 1)   # Output : action

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        action = torch.tanh(self.fc3(state)) * 2

        return action

# Add noise to Action
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def train(Buffer, Q, Q_target, P, P_target, Q_optimizer, P_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)

    # Type casting & fit dimension
    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim=1)
    rewards = torch.unsqueeze(rewards, dim=1)

    loss_Q, loss_P = 0, 0
    with torch.no_grad():
        acts = P_target(next_states)
        y = rewards + gamma*Q_target(next_states, acts) * (1 - terminateds)

    loss_Q = ((y - Q(states, actions)) ** 2).mean()

    # Update Q network
    Q_optimizer.zero_grad()
    loss_Q.backward()
    Q_optimizer.step()

    loss_P = Q(states, P(states)).mean() * (-1) # multiply -1 for converting GD to GA

    # Freezing Q Network
    for p in Q.parameters():
        p.require_grads = False

    # Update P Network
    P_optimizer.zero_grad()
    loss_P.backward()
    P_optimizer.step()

    # Unfreezing Q Network
    for p in Q.parameters():
        p.require_grads = True

    # Soft update (not periodically update, instead soft update !!)
    soft_update(Q, Q_target, P, P_target)

def soft_update(Q, Q_target, P, P_target):
    for param, target_param in zip(Q.parameters(), Q_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

    for param, target_param in zip(P.parameters(), P_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

# define hyperparameters & networks

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Q
Q = QNetwork().to(device)
Q_target = QNetwork().to(device)
Q_learning_rate = 0.001
Q_optimizer = optim.Adam(Q.parameters(), lr = Q_learning_rate)

# P
P = PNetwork().to(device)
P_target = PNetwork().to(device)
P_learning_rate = 0.0001
P_optimizer = optim.Adam(P.parameters(), lr = P_learning_rate)

# ReplayBuffer & noise
Buffer = ReplayBuffer()
noise  = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

# Training Hyperparameters
MAX_EPISODE = 200
MAX_STEP    = 1000
batch_size  = 32
gamma       = 0.99
tau         = 0.001

# For saving learning graphs
X = np.arange(0, MAX_EPISODE, 1) # 0 ~ MAX_EPISIODE까지 1 단위로 찍어보자
Y = []

# Define gym environment
#env = gym.make('Pendulum-v1', render_mode = 'human')  # Rendering = On
env = gym.make('Pendulum-v1')                          # Rendering = Off

# main
for episode in range(MAX_EPISODE):
    state, _ = env.reset()
    state    = torch.tensor(state).to(device)

    total_reward = 0
    terminated = False
    truncated  = False

    for step in range(MAX_STEP):

        action = P(state) + noise()[0]
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy())
        next_state = torch.tensor(next_state).to(device)

        total_reward += reward
        Buffer.put([state, action, reward, next_state, terminated, truncated])

        if Buffer.size() > 50:
            train(Buffer, Q, Q_target, P, P_target, Q_optimizer, P_optimizer)  # Soft update in here, too

        if terminated or truncated:
            break

        state = next_state

    Y.append(total_reward)
    print(f"# - Episode : {episode} | Total reward : {total_reward} - #")
env.close()

plt.plot(X, Y)
plt.savefig('./DDPG_test_2.png')

Y = np.array(Y)
np.save('./DDPG_2_X', X)
np.save('./DDPG_2_Y', Y)