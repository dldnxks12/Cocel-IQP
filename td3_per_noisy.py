import os
import sys
import math
import time
import random
import collections
import numpy as np
import environment
import gymnasium as gym

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque, namedtuple
from NoisyNet import *
from per import ProportionalPrioritizedMemory
from per_simple import Memory

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc_s   = NoisyLayer(15, 64)
        self.fc_a   = NoisyLayer(1, 64)

        self.fc1    = NoisyLayer(128, 256)
        self.fc2    = NoisyLayer(256, 128)
        self.fc3    = NoisyLayer(128, 64)
        self.fc_out = NoisyLayer(64, 1)

    def forward(self, state, action):
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))

        concatenate = torch.cat([h1, h2], dim = -1)

        q  = F.relu(self.fc1(concatenate))
        q  = F.relu(self.fc2(q))
        q  = F.relu(self.fc3(q))

        return self.fc_out(q)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = NoisyLayer(15, 128) # Input : State 3
        self.fc2 = NoisyLayer(128, 128)
        self.fc3 = NoisyLayer(128, 64)
        self.fc4 = NoisyLayer(64, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        state  = F.relu(self.fc3(state))
        action = torch.tanh(self.fc4(state)) # torque range : [-1 ~ 1]

        return action

def train(per_memory, Q1, Q1_target, Q2, Q2_target, P, P_target, Q1_optimizer, Q2_optimizer, P_optimizer):
    indexes, weights, experiences = per_memory.sample(batch_size)

    batch   = Experience(*zip(*experiences))
    weights = torch.tensor(weights).to(device)

    states      = [state.cpu().numpy() for state in batch.state]
    actions     = [action.cpu().numpy() for action in batch.action]
    rewards     = np.array(batch.reward, dtype = np.float32)
    next_states = [next_state.cpu().numpy() for next_state in batch.next_state]
    terminateds = np.array(batch.terminated, dtype=np.float32)
    truncateds  = np.array(batch.truncated, dtype=np.float32)

    states      = torch.tensor(states, device=device)
    actions     = torch.tensor(actions, device=device)
    rewards     = torch.tensor(rewards, device=device)
    next_states = torch.tensor(next_states, device=device)
    terminateds = torch.tensor(terminateds, device=device)
    truncateds  = torch.tensor(truncateds, device=device)

    loss_Q1, loss_Q2, loss_P = 0, 0, 0

    # Add Noise & clamping
    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim=1)
    rewards = torch.unsqueeze(rewards, dim=1)

    with torch.no_grad():

        # noise를 더한 action이 -2 ~ 2 사이를 벗어나지 않도록 clamping 처리
        acts = P_target(next_states)

        Q1_value = Q1_target(next_states, acts)
        Q2_value = Q2_target(next_states, acts)

        y = rewards + (gamma * torch.minimum(Q1_value, Q2_value) * (1 - terminateds))

    q1_value = Q1(states, actions)
    q2_value = Q2(states, actions)

    delta1 = q1_value - y
    delta2 = q2_value - y

    delta = torch.maximum(delta1, delta2)

    per_memory.update(indexes, delta.tolist())

    loss_Q1 = (delta1.pow(2) * weights).mean()
    loss_Q2 = (delta2.pow(2) * weights).mean()

    # Update Q network
    Q1_optimizer.zero_grad()
    loss_Q1.backward()
    Q1_optimizer.step()

    Q2_optimizer.zero_grad()
    loss_Q2.backward()
    Q2_optimizer.step()

    # Q1 or Q2 둘 중 아무거나 써도 상관 X
    loss_P = Q1(states, P(states)).mean() * (-1)  # multiply -1 for converting GD to GA

    # Freezing Q Network
    for p, q in zip(Q1.parameters(), Q2.parameters()):
        p.require_grads = False
        q.require_grads = False

    # Update P Network
    P_optimizer.zero_grad()
    loss_P.backward()
    P_optimizer.step()

    # Unfreezing Q Network
    for p, q in zip(Q1.parameters(), Q2.parameters()):
        p.require_grads = True
        q.require_grads = True

    # Soft update (not periodically update, instead soft update !!)
    soft_update(Q1, Q1_target, Q2, Q2_target, P, P_target)

def soft_update(Q1, Q1_target, Q2, Q2_target, P, P_target):
    for param, target_param in zip(Q1.parameters(), Q1_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

    for param, target_param in zip(Q2.parameters(), Q2_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

    for param, target_param in zip(P.parameters(), P_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

lr_pi = 0.0001 # Learning rate
lr_q  = 0.001
tau   = 0.001  # Soft update rate
gamma = 0.99  # Discount Factor
batch_size = 64


# PER Setting
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'terminated', 'truncated'])
alpha = 0.6
beta  = 0.4
per_memory = Memory(capacity = 5000)

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

xml_file = os.getcwd()+"/environment/assets/inverted_triple_pendulum.xml"
env = gym.make("InvertedTriplePendulum-v4", model_path=xml_file)
env_visual = gym.make("InvertedTriplePendulum-v4", render_mode="human", model_path=xml_file)
current_env = env


MAX_EPISODE   = 20000
max_time_step = env._max_episode_steps

for episode in range(MAX_EPISODE):

    if episode >= 15000:
        current_env = env_visual # visualization

    state, _ = current_env.reset()
    state    = torch.tensor(state).float().to(device)

    terminated, truncated = False, False
    total_reward = 0

    # Generate Episodes ...
    for time_step in range(max_time_step):
        if episode < 3:
            action = current_env.action_space.sample()
            action = torch.FloatTensor(action).to(device)
        else:
            with torch.no_grad():
                action = Pi(state)

        next_state, reward, terminated, truncated, _ = current_env.step(action.cpu().detach().numpy())
        next_state = torch.tensor(next_state).float().to(device)

        terminated = not terminated if time_step == max_time_step - 1 else terminated

        per_memory.add(Experience(state, action, reward, next_state, terminated, truncated))
        total_reward += reward

        if len(per_memory) > 100:
            train(per_memory, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer)

        if terminated or truncated:
            break

        state = next_state
    print(f"Episode : {episode} | TReward : {total_reward}")

current_env.close()

