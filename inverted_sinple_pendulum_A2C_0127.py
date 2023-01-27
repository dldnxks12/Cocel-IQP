# 1월 27일 기준 택도 없음
# Reward가 올라갔다가도 금방 바닥으로 떨어짐 
# 카이스트에서 진행헀던 OpenAI Pendulum 문제는 DDPG로 얼추 풀었고, TD3로 확실히 해결했었다. 
# 위 알고리즘으로 재

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
        self.fc1   = nn.Linear(7, 128)  # Input : state
        self.fc2   = nn.Linear(128, 64)  # Output : Softmax policy for action distribution
        self.fc_mu = nn.Linear(64, 1)  # Output : Softmax policy for action distribution

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = torch.tanh(self.fc_mu(x))

        return mu # action [-1 ~ 1]

# Critic
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 128)  # Input : state
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)   # Output : Appoximated V Value

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        Approxed_value = self.fc3(x)

        return Approxed_value

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
    #for target_param, current_param in zip(Target_Network.parameters(), Current_Network.parameters()):
    #   target_param.data.copy_(target_param.data * (1.0 - beta) + current_param.data * beta)
    Target_Network.load_state_dict(Current_Network.state_dict())

def train(memory, V, V_target, V_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = memory.sample(64)

    critic_loss = 0
    actor_loss  = 0

    # Critic Update
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        if terminated or truncated:
            y = reward
        else:
            with torch.no_grad(): # No gradient update while simple inference...
                y = reward + gamma * V_target(next_state)

        critic_loss += (y - V(state))**2

    critic_loss = critic_loss / 64

    V_optimizer.zero_grad()
    critic_loss.backward()
    V_optimizer.step()

    # Actor Update
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        actor_loss += (reward + (gamma * V(next_state)) - V(state)) * (pi(state).log())

    actor_loss = -actor_loss / 64 # Gradient Ascent

    pi_optimizer.zero_grad()
    actor_loss.backward()
    pi_optimizer.step()

xml_file = os.getcwd()+"/environment/assets/inverted_single_pendulum.xml"
env      = gym.make("InvertedSinglePendulum-v4", model_path=xml_file)
env_eval = gym.make("InvertedSinglePendulum-v4", render_mode="human", model_path=xml_file)

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
alpha = 0.001
beta  = 0.01 # Update Weight
gamma = 0.99
episode = 0
MAX_EPISODE = 10000

pi = PolicyNetwork().to(device)
V  = ValueNetwork().to(device)
V_target = ValueNetwork().to(device)

V_target.load_state_dict(V.state_dict())

pi_optimizer = optim.Adam(pi.parameters(), lr = alpha)
V_optimizer = optim.Adam(V.parameters(), lr = alpha)

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
        next_state = torch.tensor(next_state).float().to(device)
        memory.put((state, action, reward, next_state, terminated, truncated))
        score += reward
        state = next_state

        if terminated or truncated:
            break

    if memory.size() > 2000:
        train(memory, V, V_target, V_optimizer)

        if episode % 10 == 0:
            soft_update(V_target, V)

    print(f"Episode : {episode} || Reward : {score} ")
    episode += 1

env.close()
