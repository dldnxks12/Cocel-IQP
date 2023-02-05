"""

tested with DDPG

rewards get increased approximately at 1000 episodes

"""

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

# Actor
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(7, 64)   # Input : state
        self.fc2   = nn.Linear(64, 64)   # Input : state
        self.fc3   = nn.Linear(64, 1)   # Output : Softmax policy for action distribution

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc3(state))

        return action

# Critic
class QValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Linear(7, 64)  # Input : state
        self.fc_a = nn.Linear(1, 64)  # Input : state
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)   # Output : Appoximated Q Value

    def forward(self, x, a): # Input : state , action
        h1  = F.relu(self.fc_s(x))
        h2  = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim = -1)

        q = F.relu(self.fc1(cat))
        return self.fc2(q)

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

def train(Q, Q_target, Pi, Pi_target, Q_optimizer, Pi_optimizer, Buffer, batch_size):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)

    Q_loss = 0

    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states,
                                                                        terminateds, truncateds):
        if terminated or truncated:
            y = reward
        else:
            next_act = Pi_target(next_state) + ou_noise()[0]
            y = reward + gamma * Q_target(next_state, next_act)

        action = action.to(device)
        Q_loss += (y - Q(state, action)) ** 2

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
    Pi_loss = -1 * (Pi_loss / batch_size)
    Pi_optimizer.zero_grad()
    Pi_loss.backward()
    Pi_optimizer.step()

    for p in Q.parameters():
        p.requires_grad = True

xml_file = os.getcwd()+"/environment/assets/inverted_single_pendulum.xml"
#env      = gym.make("InvertedSinglePendulum-v4", render_mode = 'human', model_path=xml_file)
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
lr_pi = 0.00005
lr_q  = 0.0005
beta  = 0.005

gamma = 0.95
episode = 0
batch_size = 64

pi = PolicyNetwork().to(device)
pi_target = PolicyNetwork().to(device)
pi_target.load_state_dict(pi.state_dict())
pi_optimizer = optim.Adam(pi.parameters(), lr = lr_pi)

Q  = QValueNetwork().to(device)
Q_target = QValueNetwork().to(device)
Q_target.load_state_dict(Q.state_dict())
Q_optimizer = optim.Adam(Q.parameters(), lr = lr_q)

ou_noise = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

MAX_EPISODE = 50000
max_time_step = 500

for episode in range(MAX_EPISODE):

    state, _ = env.reset() # state -> numpy
    state = torch.tensor(state).float().to(device)
    score = 0

    terminated = False
    truncated  = False

    for time_step in range(max_time_step):
        with torch.no_grad():
            action = (pi(state) + ou_noise()[0]).cpu().detach().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Type casting
        next_state = torch.tensor(next_state).float().to(device)
        action = torch.tensor(action).float().to(device)

        score += reward

        # Replay buffer
        memory.put([state, action, reward, next_state, terminated, truncated])

        if memory.size() > 2000:
            train(Q, Q_target, pi, pi_target, Q_optimizer, pi_optimizer, memory, batch_size)
            if time_step % 5 == 0:
                soft_update(Q_target, Q)
                soft_update(pi_target, pi)

        if terminated or truncated:
            break

        state = next_state

    print(f"Episode : {episode} || Reward : {score} ")

env.close()