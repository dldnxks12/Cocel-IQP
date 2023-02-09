# action : 1 continuous action : force (-1 ~ 1)
# state  : 11 continuous state  : cart pos , pole1_sin , pole1_cos, pole2_sin , pole2_cos, cart vel, pole_vel1, pole_vel2, contraint-1, constraint-2 constraint-3

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

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc_s = nn.Linear(11, 64)
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
        self.fc1 = nn.Linear(11, 64) # Input : State 3
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc3(state))

        return action

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
        terminateds = torch.tensor(terminateds, device=device, dtype=torch.float)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

def train(time_step, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer, Buffer, batch_size):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)
    Q1_loss, Q2_loss, pi_loss = 0, 0, 0

    noise_bar = torch.unsqueeze(torch.clamp(torch.randn_like(actions) * 0.2, -0.5, 0.5), dim = 1)
    action_bar = torch.clamp(Pi_target(next_states) + noise_bar, -1, 1) # next_state : 32x3 , action_bar : 32 x 1

    q1_value = Q1_target(next_states, action_bar)
    q2_value = Q2_target(next_states, action_bar)

    rewards = torch.unsqueeze(rewards, dim = 1)
    terminateds = torch.unsqueeze(terminateds, dim = 1)

    y = rewards + ( gamma * torch.minimum(q1_value, q2_value) * (1 - terminateds)) # minimum loss value for update

    actions = torch.unsqueeze(actions, dim = 1)
    Q1_loss = F.mse_loss(Q1(states, actions), y.detach()) # Q1 Network Update
    Q1_optimizer.zero_grad()
    Q1_loss.backward()
    Q1_optimizer.step()

    Q2_loss = F.mse_loss(Q2(states, actions), y.detach()) # Q2 Network Update
    Q2_optimizer.zero_grad()
    Q2_loss.backward()
    Q2_optimizer.step()

    # Periodically update this
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

xml_file = os.getcwd()+"/environment/assets/inverted_double_pendulum.xml"
env = gym.make("InvertedDoublePendulum-v4",  model_path=xml_file)
env_eval = gym.make("InvertedDoublePendulum-v4", render_mode="human", model_path=xml_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

lr_pi = 0.00005 # Learning rate
lr_q  = 0.0005
tau   = 0.005  # Soft update rate
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


MAX_EPISODE   = 3000
max_time_step = env._max_episode_steps

Buffer = ReplayBuffer()
noise  = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

for episode in range(MAX_EPISODE):

    if episode % 50 == 0:
        state, _ = env_eval.reset()
        state = torch.tensor(state).float().to(device)

        terminated, truncated = False, False
        total_reward = 0

        # Generate Episodes ...
        for time_step in range(max_time_step):

            with torch.no_grad():
                action = (Pi(state) + noise()[0]).cpu()

            # Reward 조작
            next_state, reward, terminated, truncated, info = env_eval.step(action)
            next_state = torch.tensor(next_state).float().to(device)
            total_reward += reward

            if terminated or truncated:
                break

            state = next_state

        print(f"Episode : {episode} | TReward : {total_reward}")

    else:
        state, _ = env.reset()
        state       = torch.tensor(state).float().to(device)

        terminated, truncated = False, False
        total_reward = 0

        # Generate Episodes ...
        for time_step in range(max_time_step):

            with torch.no_grad():
                action = (Pi(state) + noise()[0]).cpu()

            next_state, reward, terminated, truncated, info = env.step(action)


            next_state = torch.tensor(next_state).float().to(device)
            terminated = not terminated if time_step == max_time_step - 1 else terminated
            Buffer.put([state, action, reward, next_state, terminated, truncated])
            total_reward += reward

            if Buffer.size() > 1000: # Train Q, Pi
                train(time_step, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer, Buffer, batch_size)

                if time_step % 5 == 0:
                    for param_target, param, param_target2, param2 in zip(Q1_target.parameters(), Q1.parameters(),
                                                                          Q2_target.parameters(), Q2.parameters()):
                        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
                        param_target2.data.copy_(param_target2.data * (1.0 - tau) + param2.data * tau)

                    for param_target, param in zip(Pi_target.parameters(), Pi.parameters()):
                        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

            if terminated or truncated:
                break

            state = next_state

        print(f"Episode : {episode} | TReward : {total_reward}")

env.close()
env_eval.close()

