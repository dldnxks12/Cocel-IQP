

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

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=20000)

    def put(self, transition):
        self.buffer.append(transition)

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

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc_s   = NoisyLayer(15, 128)
        self.fc_a   = NoisyLayer(1, 128)

        self.fc1    = NoisyLayer(256, 256)
        self.fc4    = NoisyLayer(256, 64)
        self.fc_out = NoisyLayer(64, 1)

    def forward(self, state, action):
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))

        concatenate = torch.cat([h1, h2], dim = -1)

        q  = F.relu(self.fc1(concatenate))
        q  = F.relu(self.fc4(q))

        return self.fc_out(q)

class PolicyNetwork1(nn.Module):
    def __init__(self):
        super(PolicyNetwork1, self).__init__()
        self.fc1 = NoisyLayer(15, 256) # Input : State 3
        self.fc2 = NoisyLayer(256, 64)
        self.fc5 = NoisyLayer(64, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc5(state)) # torque range : [-1 ~ 1]

        return action

class PolicyNetwork2(nn.Module):
    def __init__(self):
        super(PolicyNetwork2, self).__init__()
        self.fc1 = NoisyLayer(15, 64) # Input : State 3
        self.fc2 = NoisyLayer(64, 64)
        self.fc5 = NoisyLayer(64, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc5(state)) # torque range : [-1 ~ 1]

        return action

class PolicyNetwork3(nn.Module):
    def __init__(self):
        super(PolicyNetwork3, self).__init__()
        self.fc1 = NoisyLayer(15, 128) # Input : State 3
        self.fc2 = NoisyLayer(128, 128)
        self.fc5 = NoisyLayer(128, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc5(state)) # torque range : [-1 ~ 1]

        return action

class PolicyNetwork4(nn.Module):
    def __init__(self):
        super(PolicyNetwork4, self).__init__()
        self.fc1 = NoisyLayer(15, 512) # Input : State 3
        self.fc2 = NoisyLayer(512, 256)
        self.fc5 = NoisyLayer(256, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc5(state)) # torque range : [-1 ~ 1]

        return action

class PolicyNetwork5(nn.Module):
    def __init__(self):
        super(PolicyNetwork5, self).__init__()
        self.fc1 = NoisyLayer(15, 256) # Input : State 3
        self.fc2 = NoisyLayer(256, 64)
        self.fc5 = NoisyLayer(64, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc5(state)) # torque range : [-1 ~ 1]

        return action

class PolicyNetwork6(nn.Module):
    def __init__(self):
        super(PolicyNetwork6, self).__init__()
        self.fc1 = NoisyLayer(15, 128) # Input : State 3
        self.fc2 = NoisyLayer(128, 32)
        self.fc5 = NoisyLayer(32, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc5(state)) # torque range : [-1 ~ 1]

        return action

def train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer, step):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)

    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim = 1)
    rewards     = torch.unsqueeze(rewards, dim = 1)

    Q1_loss, Q2_loss, pi_loss = 0, 0, 0

    with torch.no_grad():
        action_bar = Pi_target(next_states)

        q1_value = Q1_target(next_states, action_bar)
        q2_value = Q2_target(next_states, action_bar)

        y = rewards + ( gamma * torch.minimum(q1_value, q2_value) * (1 - terminateds))

    Q1_loss = ( (y - Q1(states, actions)) ** 2 ).mean()
    Q2_loss = ( (y - Q2(states, actions)) ** 2 ).mean()

    Q1_optimizer.zero_grad()
    Q1_loss.backward()
    Q1_optimizer.step()

    Q2_optimizer.zero_grad()
    Q2_loss.backward()
    Q2_optimizer.step()

    # Delayed Policy Update
    if step % 2 == 0:
        for p, q in zip(Q1.parameters(), Q2.parameters()):
            p.requires_grad = False
            q.requires_grad = False

        pi_loss = - Q1(states, Pi(states)).mean()

        Pi_optimizer.zero_grad()
        pi_loss.backward()
        Pi_optimizer.step()

        for p, q in zip(Q1.parameters(), Q2.parameters()):
            p.requires_grad = True
            q.requires_grad = True

        # Soft update (not periodically update, instead soft update !!)
        soft_update(Q1, Q1_target, Q2, Q2_target, Pi, Pi_target)

def soft_update(Q1, Q1_target, Q2, Q2_target, P, P_target):
    for param, target_param in zip(Q1.parameters(), Q1_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

    for param, target_param in zip(Q2.parameters(), Q2_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

    for param, target_param in zip(P.parameters(), P_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

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
gamma = 0.99  # Discount Factor
batch_size = 256


# PER Setting
Buffer = ReplayBuffer()

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
Pi1 = PolicyNetwork1().to(device)
Pi2 = PolicyNetwork2().to(device)
Pi3 = PolicyNetwork3().to(device)
Pi4 = PolicyNetwork4().to(device)
Pi5 = PolicyNetwork5().to(device)
Pi6 = PolicyNetwork6().to(device)

Pi_target1 = PolicyNetwork1().to(device)
Pi_target2 = PolicyNetwork2().to(device)
Pi_target3 = PolicyNetwork3().to(device)
Pi_target4 = PolicyNetwork4().to(device)
Pi_target5 = PolicyNetwork5().to(device)
Pi_target6 = PolicyNetwork6().to(device)
Pi_optimizer1 = optim.Adam(Pi1.parameters(), lr = lr_pi)
Pi_optimizer2 = optim.Adam(Pi2.parameters(), lr = lr_pi)
Pi_optimizer3 = optim.Adam(Pi3.parameters(), lr = lr_pi)
Pi_optimizer4 = optim.Adam(Pi4.parameters(), lr = lr_pi)
Pi_optimizer5 = optim.Adam(Pi5.parameters(), lr = lr_pi)
Pi_optimizer6 = optim.Adam(Pi6.parameters(), lr = lr_pi)
Pi_target1.load_state_dict(Pi1.state_dict())
Pi_target2.load_state_dict(Pi2.state_dict())
Pi_target3.load_state_dict(Pi3.state_dict())
Pi_target4.load_state_dict(Pi4.state_dict())
Pi_target5.load_state_dict(Pi5.state_dict())
Pi_target6.load_state_dict(Pi6.state_dict())

xml_file = os.getcwd()+"/environment/assets/inverted_triple_pendulum.xml"
env = gym.make("InvertedTriplePendulum-v4", model_path=xml_file)
env_visual = gym.make("InvertedTriplePendulum-v4", render_mode="human", model_path=xml_file)
current_env = env
noise  = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

MAX_EPISODE   = 2000
max_time_step = env._max_episode_steps
X = np.arange(0, MAX_EPISODE, 1)
Y = []

success_counter = 0
softmax_recorder = []
for episode in range(MAX_EPISODE):

    state, _ = current_env.reset()
    state    = torch.tensor(state).float().to(device)

    terminated, truncated = False, False
    total_reward = 0

    # Generate Episodes ...
    for time_step in range(max_time_step):
        if episode < 2:
            action = current_env.action_space.sample()
            action = torch.FloatTensor(action).to(device)

        else:
            noise_idx = np.random.choice([0, 1, 2, 3, 4, 5], 2, replace=False)
            noise_list = [0, 0, 0, 0, 0, 0]

            noise_list[noise_idx[0]] = noise()[0]
            noise_list[noise_idx[1]] = noise()[0]

            noise1 = noise_list[0]
            noise2 = noise_list[1]
            noise3 = noise_list[2]
            noise4 = noise_list[3]
            noise5 = noise_list[4]
            noise6 = noise_list[5]

            with torch.no_grad():
                action1 = torch.clamp((Pi1(state) + noise1), -1, 1)
                action2 = torch.clamp((Pi2(state) + noise2), -1, 1)
                action3 = torch.clamp((Pi3(state) + noise3), -1, 1)
                action4 = torch.clamp((Pi4(state) + noise4), -1, 1)
                action5 = torch.clamp((Pi5(state) + noise5), -1, 1)
                action6 = torch.clamp((Pi6(state) + noise6), -1, 1)

                # Q1 기준으로 해보자
                eval_q1 = Q1(state, action1)
                eval_q2 = Q1(state, action2)
                eval_q3 = Q1(state, action3)
                eval_q4 = Q1(state, action4)
                eval_q5 = Q1(state, action5)
                eval_q6 = Q1(state, action6)

            # Stack actions
            evaluations = torch.stack([eval_q1, eval_q2, eval_q3, eval_q4, eval_q5, eval_q6])
            action_softmax = torch.nn.functional.softmax(evaluations, dim=0).squeeze()

            if time_step % 100 == 0:
                softmax_recorder.append(action_softmax.cpu().detach().numpy())

            # Select Action according to softmaxed evaluations
            action_list = [action1, action2, action3, action4, action5, action6]
            action_index = [0, 1, 2, 3, 4, 5]

            # Soft Voting
            choice_action = np.random.choice(action_index, 1, p=action_softmax.cpu().detach().numpy())
            action = action_list[choice_action[0]]

        next_state, reward, terminated, truncated, _ = current_env.step(action.cpu().detach().numpy())
        next_state = torch.tensor(next_state).float().to(device)

        terminated = not terminated if time_step == max_time_step - 1 else terminated

        Buffer.put([state, action, reward, next_state, terminated, truncated])
        total_reward += reward

        if Buffer.size() > 2000:  # Train Q, Pi
            for _ in range(3):
                train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi1, Pi_target1, Q1_optimizer, Q2_optimizer, Pi_optimizer1, time_step)
                train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi2, Pi_target2, Q1_optimizer, Q2_optimizer, Pi_optimizer2, time_step)
                train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi3, Pi_target3, Q1_optimizer, Q2_optimizer, Pi_optimizer3, time_step)
                train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi4, Pi_target4, Q1_optimizer, Q2_optimizer, Pi_optimizer4, time_step)
                train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi5, Pi_target5, Q1_optimizer, Q2_optimizer, Pi_optimizer5, time_step)
                train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi6, Pi_target6, Q1_optimizer, Q2_optimizer, Pi_optimizer6, time_step)

        if terminated or truncated:
            break

        state = next_state

    if total_reward > 550:
        current_env = env_visual

    Y.append(total_reward)
    print(f"Episode : {episode} | TReward : {total_reward}")

    if episode % 10 == 0:
        Y = np.array(Y)
        np.save('./TD3_noisy_ensemble_0401_Y1', Y)
        np.save("./Softmax_0401_1", softmax_recorder)
        Y = list(Y)

current_env.close()




