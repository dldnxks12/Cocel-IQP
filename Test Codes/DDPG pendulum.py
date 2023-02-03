# DDPG for Episodic case

"""

state
    - x ( cos(theta) )
    - y ( sin(theta) )
    - a (angular velocity)

action
    - Torque, [-2.0 , 2.0] - 1D Continuous Value

reward function

     - theta2 + 0.1 * theta_dt2 + 0.001 * torque2

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
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

def train(Q, Q_target, Pi, Pi_target, Q_optimizer, Pi_optimizer, Buffer, batch_size):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)

    Q_loss = 0

    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):

        if terminated or truncated:
            y = reward
        else:
            next_act = Pi_target(next_state) + noise()[0]
            y = reward + gamma * Q_target(next_state, next_act)

        action = action.to(device)
        Q_loss += (y - Q(state, action))**2

    # Gradient descent
    Q_loss = Q_loss / batch_size
    Q_optimizer.zero_grad()
    Q_loss.backward()
    Q_optimizer.step()

    Pi_loss = 0

    for p in Q.parameters():
        p.requires_grad = False

    for state in states:
        Pi_loss += Q(state, Pi(state))

    # Gradient Ascent
    Pi_loss = -1 * (Pi_loss/batch_size)
    Pi_optimizer.zero_grad()
    Pi_loss.backward()
    Pi_optimizer.step()

    for p in Q.parameters():
        p.requires_grad = True

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

lr_pi = 0.0001 # Learning rate
lr_q  = 0.001
tau   = 0.001  # Soft update rate
gamma = 0.95  # Discount Factor
batch_size = 32

# Q function
Q = QNetwork().to(device)
Q_optimizer = optim.Adam(Q.parameters(), lr = lr_q)
Q_target = QNetwork().to(device)
Q_target.load_state_dict(Q.state_dict())

# Policy
Pi = PolicyNetwork().to(device)
Pi_target = PolicyNetwork().to(device)
Pi_optimizer = optim.Adam(Pi.parameters(), lr = lr_pi)
Pi_target.load_state_dict(Pi.state_dict())

# Freezing Target Parameters
for p in Q_target.parameters():
    p.requires_grad = False

for m in Pi_target.parameters():
    m.requires_grad = False

env = gym.make('Pendulum-v1', g=9.81)

max_time_step = 1000
MAX_EPISODE   = 1000

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

        if terminated or truncated:
            break

        state = next_state

        if Buffer.size() > 2000: # Train Q, Pi
            train(Q, Q_target, Pi, Pi_target, Q_optimizer, Pi_optimizer, Buffer, batch_size)

            if time_step % 5 == 0: # Soft update
                for param_target, param in zip(Q_target.parameters(), Q.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

                for param_target, param in zip(Pi_target.parameters(), Pi.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    print(f"Episode : {episode} | TReward : {total_reward}")

env.close()