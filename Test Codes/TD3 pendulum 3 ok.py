"""
# TD3
https://github.com/sfujim/TD3/blob/master/TD3.py <- 보고 고친 코드

Critic -> Q1, Q2 둘 다 업데이트 해야한다.

Actor  -> Q1 or Q2 둘 중 아무거나 써도 된다.
"""

import os
import sys
import math
import random
import collections
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)

        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1) # Output : Q Value

    def forward(self, state, action):
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))

        cat = torch.cat([h1, h2], dim = -1)
        q = F.relu(self.fc_q(cat))

        return self.fc_out(q)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64) # Input : State 3
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc3(state)) * 2

        return action # [-2, 2]

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n_sample):
        mini_batch = random.sample(self.buffer, n_sample)

        states, actions, rewards, next_states, terminateds, truncateds = [],[],[],[],[],[]
        for state, action, reward, next_state, terminated, truncated in mini_batch:

            states.append(state.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.cpu().numpy())
            terminateds.append(terminated)
            truncateds.append(truncated)

        states = torch.tensor(states, device = device, dtype = torch.float)
        actions = torch.tensor(actions, device = device, dtype = torch.float)
        next_states = torch.tensor(next_states, device = device, dtype = torch.float)
        rewards     = torch.tensor(rewards, device = device, dtype = torch.float)
        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

def train(time_step, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer, Buffer, batch_size):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)
    Q1_loss, Q2_loss, pi_loss = 0, 0, 0

    # Action + noise clamping (min_action ~ max_action)
    noise_bar  = torch.tensor(noise()[0])
    action_bar = torch.clamp(Pi_target(next_states) + noise_bar, -2, 2) # next_state : 32x3 , action_bar : 32 x 1

    q1_value = Q1_target(next_states, action_bar)
    q2_value = Q2_target(next_states, action_bar)

    q1_mean = q1_value.mean()
    q2_mean = q2_value.mean()
    selected_Q_idx = torch.argmin(torch.tensor([q1_mean, q2_mean]), axis = 0)
    q_list = [q1_value, q2_value]
    dones = []

    for terminated, truncated in zip(terminateds, truncateds):
        if (terminated == True) or (truncated == True):
            dones.append([0])
        else:
            dones.append([1])

    dones = torch.tensor(dones, device = device)
    actions = torch.unsqueeze(actions, dim = 1)
    rewards = torch.unsqueeze(rewards, dim = 1)

    y = rewards + ( gamma * q_list[selected_Q_idx] * dones ) # minimum loss value for update

    Q1_loss = F.mse_loss(Q1(states, actions), y.detach()) # Q1 Network Update
    Q1_optimizer.zero_grad()
    Q1_loss.backward()
    Q1_optimizer.step()

    Q2_loss = F.mse_loss(Q2(states, actions), y.detach()) # Q2 Network Update
    Q2_optimizer.zero_grad()
    Q2_loss.backward()
    Q2_optimizer.step()

    # Periodically update this
    if time_step % 5 == 0:  # Soft update
        for p, q in zip(Q1.parameters(), Q2.parameters()):
            p.requires_grad = False
            q.requires_grad = False

        pi_loss = -Q1(states, Pi(states)).mean()

        Pi_optimizer.zero_grad()
        pi_loss.backward()
        Pi_optimizer.step()

        for p, q in zip(Q1.parameters(), Q2.parameters()):
            p.requires_grad = True
            q.requires_grad = True

        for param_target, param, param_target2, param2 in zip(Q1_target.parameters(), Q1.parameters(),
                                                              Q2_target.parameters(), Q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            param_target2.data.copy_(param_target2.data * (1.0 - tau) + param2.data * tau)

        for param_target, param in zip(Pi_target.parameters(), Pi.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


# Add Noise to deterministic action for improving exploration property
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

lr_pi = 0.0009 # Learning rate
lr_q  = 0.009
tau   = 0.009  # Soft update rate
gamma = 0.95  # Discount Factor
batch_size = 64

# Q function
Q1 = QNetwork().to(device)
Q2 = QNetwork().to(device)

Q1_optimizer = optim.Adam(Q1.parameters(), lr = lr_q)
Q2_optimizer = optim.Adam(Q2.parameters(), lr = lr_q)

Q1_target = QNetwork().to(device)
Q1_target.load_state_dict(Q1.state_dict())

Q2_target = QNetwork().to(device)
Q2_target.load_state_dict(Q2.state_dict())

# Policy
Pi = PolicyNetwork().to(device)
Pi_target = PolicyNetwork().to(device)
Pi_optimizer = optim.Adam(Pi.parameters(), lr = lr_pi)
Pi_target.load_state_dict(Pi.state_dict())

# Freezing Target Parameters
for p, q in zip(Q1_target.parameters(), Q2_target.parameters()):
    p.requires_grad = False
    q.requires_grad = False

for m in Pi_target.parameters():
    m.requires_grad = False

env = gym.make('Pendulum-v1', g=9.81)

MAX_EPISODE   = 1000
max_time_step = 500

Buffer = ReplayBuffer()
noise  = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

for episode in range(MAX_EPISODE):

    observation = env.reset()[0]
    state       = torch.tensor(observation).to(device)

    terminated, truncated = False, False
    total_reward = 0

    # Generate Episodes ...
    for time_step in range(max_time_step):

        with torch.no_grad():
            action = (Pi(state) + noise()[0]).cpu()

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(next_observation).to(device)

        Buffer.put([state, action, reward, next_state, terminated, truncated])
        total_reward += reward

        if Buffer.size() > 2000: # Train Q, Pi
            train(time_step, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer, Buffer, batch_size)

        if terminated or truncated:
            break

        state = next_state

    print(f"Episode : {episode} | TReward : {total_reward}")

env.close()