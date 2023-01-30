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
        self.fc1   = nn.Linear(7, 64)   # Input : state
        self.fc2 = nn.Linear(64, 128)   # Input : state
        self.fc3   = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)   # Output : Softmax policy for action distribution

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        mu = torch.tanh(self.fc_mu(x))

        return mu # action [-1 ~ 1]

# Critic
class QValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Linear(7, 64)  # Input : state
        self.fc_a = nn.Linear(1, 64)  # Input : action

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)   # Output : Appoximated Q Value

    def forward(self, x, a): # Input : state , action

        sta = self.fc_s(x)
        sta = F.relu(sta)

        act = self.fc_a(a)
        act = F.relu(act)

        cat = torch.cat([sta, act], dim = 0)

        q = F.relu(self.fc2(cat))
        q = F.relu(self.fc3(q))
        q = self.fc4(q)

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

def train(memory, Q, Q_target, Q_optimizer, pi, pi_target, pi_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = memory.sample(batch_size)

    critic_loss = 0
    actor_loss  = 0

    # Critic Update
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        if terminated or truncated:
            y = reward
        else:
            with torch.no_grad(): # Target Network의 gradient 계산 X
                y = reward + (gamma * Q_target(next_state, pi_target(next_state)))
        critic_loss += (y - Q(state, action))**2
    critic_loss = critic_loss / batch_size

    Q_optimizer.zero_grad()
    critic_loss.backward()
    Q_optimizer.step()

    # Actor Update
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        actor_loss += Q(state, pi(state))

    actor_loss = -(actor_loss / batch_size) # Gradient Ascent

    pi_optimizer.zero_grad()
    actor_loss.backward()
    pi_optimizer.step()

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
lr_pi = 0.0005
lr_q  = 0.001
beta  = 0.005 # Update Weight
gamma = 0.99
episode = 0
batch_size = 32
MAX_EPISODE = 10000

pi = PolicyNetwork().to(device)
pi_target = PolicyNetwork().to(device)

Q  = QValueNetwork().to(device)
Q_target = QValueNetwork().to(device)

Q_target.load_state_dict(Q.state_dict())
pi_target.load_state_dict(pi.state_dict())

pi_optimizer = optim.Adam(pi.parameters(), lr = lr_pi)
Q_optimizer = optim.Adam(Q.parameters(), lr = lr_q)

ou_noise = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

while episode < MAX_EPISODE:

    # Initial Set up for training ..

    state, _ = env.reset() # state -> numpy
    state = torch.tensor(state).float().to(device)
    score = 0

    terminated = False
    truncated  = False

    while True:
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
        train(memory, Q, Q_target, Q_optimizer, pi, pi_target, pi_optimizer)
        soft_update(Q_target, Q)
        soft_update(pi_target, pi)

    print(f"Episode : {episode} || Reward : {score} ")
    episode += 1

env.close()
