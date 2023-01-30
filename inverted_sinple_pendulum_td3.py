import os
import sys
import math
import random
import collections
import numpy as np
import environment
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

# action : 1 continuous action : force (-1 ~ 1)
# state  : 7 continuous state  : cart pos , pole_sin , pole_cos, cart vel, pole_vel, contraint-1, constraint-2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Currently Working on {device}")


# 1 Actor / 1 Critic
# Replay Buffer / Soft Update
# Exploration Noise

# Actor
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(7, 256)   # Input : state
        self.fc2   = nn.Linear(256, 256)   # Input : state
        self.fc_mu = nn.Linear(256, 1)   # Output : Softmax policy for action distribution

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = torch.tanh(self.fc_mu(x))
        return mu # action [-1 ~ 1]

# Critic
class QValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Linear(7, 128)  # Input : state
        self.fc_a = nn.Linear(1, 128)  # Input : state
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)   # Output : Appoximated Q Value

    def forward(self, x, a): # Input : state , action
        h1  = F.relu(self.fc_s(x))
        h2  = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim = 0)

        q = F.relu(self.fc1(cat))
        q = self.fc2(q)
        return q

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen = 50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, number):
        mini_batch = random.sample(self.buffer, number)
        states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, terminated, truncated = transition

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

def soft_update(Target_Network, Current_Network):
    for target_param, current_param in zip(Target_Network.parameters(), Current_Network.parameters()):
       target_param.data.copy_(target_param.data * (1.0 - beta) + current_param.data * beta)
    # Target_Network.load_state_dict(Current_Network.state_dict())

def train(memory, Q1, Q2, Q1_target, Q2_target, Q1_optimizer, Q2_optimizer, pi, pi_target, pi_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = memory.sample(batch_size)

    Q_loss, pi_loss = 0, 0
    q_value_idx = 0
    noise_bar = torch.clamp(torch.tensor(ou_noise()[0]), -1, 1)
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        if terminated or truncated:
            y = reward

        else:
            with torch.no_grad():
                action_bar = pi_target(next_state) + noise_bar
                q1_value = Q1_target(next_state, action_bar)
                q2_value = Q2_target(next_state, action_bar)

                q_value_idx = torch.argmin(torch.tensor([q1_value, q2_value]), axis = 0 )
                if q_value_idx == 0:
                    y = reward + gamma * Q2_target(next_state, action_bar)
                else:
                    y = reward + gamma * Q1_target(next_state, action_bar)

        if q_value_idx == 0:
            Q_loss = (y - Q1_target(state, action)) ** 2
        else:
            Q_loss = (y - Q2_target(state, action)) ** 2

    Q_loss /= batch_size

    if q_value_idx == 0:
        Q1_optimizer.zero_grad()
        Q_loss.backward()
        Q1_optimizer.step()

    else:
        Q2_optimizer.zero_grad()
        Q_loss.backward()
        Q2_optimizer.step()

    if q_value_idx == 0:
        for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
            pi_loss -= Q1(state, pi(state))
    else:
        for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
            pi_loss -= Q2(state, pi(state))

        pi_loss /= batch_size
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step


xml_file = os.getcwd()+"/environment/assets/inverted_single_pendulum.xml"
env      = gym.make("InvertedSinglePendulum-v4", model_path=xml_file)

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

memory= ReplayBuffer()
lr_pi = 0.0001
lr_q  = 0.0001
beta  = 0.005 # Update Weight
gamma = 0.99
episode = 0
batch_size = 32
MAX_EPISODE = 10000

pi = PolicyNetwork().to(device)
pi_target = PolicyNetwork().to(device)
pi_target.load_state_dict(pi.state_dict())
pi_optimizer = optim.Adam(pi.parameters(), lr = lr_pi)

Q1  = QValueNetwork().to(device)
Q1_target = QValueNetwork().to(device)
Q1_target.load_state_dict(Q1.state_dict())
Q1_optimizer = optim.Adam(Q1.parameters(), lr = lr_q)

Q2  = QValueNetwork().to(device)
Q2_target = QValueNetwork().to(device)
Q2_target.load_state_dict(Q2.state_dict())
Q2_optimizer = optim.Adam(Q2.parameters(), lr = lr_q)

ou_noise = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

while episode < MAX_EPISODE:

    state, _ = env.reset() # state -> numpy
    state = torch.tensor(state).float().to(device)
    score = 0

    terminated = False
    truncated  = False

    while True:

        # action + exploration with Orne Noise
        action = (pi(state) + ou_noise()[0]).cpu().detach().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Type casting
        next_state = torch.tensor(next_state).float().to(device)
        action = torch.tensor(action).float().to(device)

        # Replay buffer
        memory.put((state, action, reward, next_state, terminated, truncated))

        score += reward
        state = next_state

        if terminated or truncated:
            break

        if memory.size() > 2000:
            train(memory, Q1, Q2, Q1_target, Q2_target, Q1_optimizer, Q2_optimizer, pi, pi_target, pi_optimizer)
            soft_update(Q1_target, Q1)
            soft_update(Q2_target, Q2)
            soft_update(pi_target, pi)

    print(f"Episode : {episode} || Reward : {score} ")
    episode += 1

env.close()
