"""

My first ditributed reinforcement learning - A3C

*reference : https://github.com/keep9oing/PG-Family/blob/main/A3C.py

"""

import sys
import time
import random
import numpy as np
import gymnasium as gym
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, namedtuple


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_space = None, num_hidden_layer = 2, hidden_dim = None):
        super(Critic, self).__init__()

        assert state_space is not None, "None state_state input : state_space should be assigned"

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(state_space, hidden_dim))

        for i in range(num_hidden_layer):
            self.layer.append(nn.Linear(hidden_dim, hidden_dim))

        self.layer.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        for layer in self.layer[:-1]:
            x = F.relu(layer(x))

        out = self.layer[-1](x)

        return out

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_space = None, action_space = None, num_hidden_layer = 2, hidden_dim = None):
        super(Actor, self).__init__()

        assert state_space is not None,  "None state_space input"
        assert action_space is not None, "None action_space input"

        if hidden_dim is None:
            hidden_dim = state_space * 2


        # define network
        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(state_space, hidden_dim))

        for i in range(num_hidden_layer):
            self.layer.append(nn.Linear(hidden_dim, hidden_dim))

        self.layer.append(nn.Linear(hidden_dim, action_space))


    def forward(self, x):
        for layer in self.layer[:-1]:
            x = F.relu(layer(x))

        out = F.softmax(self.layer[-1](x), dim = 0)

        return out

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

def seed_torch(seed):
    torch.manual_seed(seed) # seed 고정
    if torch.backends.cudnn.enabled == True: # cudnn 사용할 시...

        # network의 입력 사이즈가 달라지지 않을 때 사용하면 좋다.
        # 그 크기에 맞는 최적의 연산 알고리즘을 골라주기 때문
        # 따라서 입력 사이즈가 바뀌면 그때마다 최적 알고리즘을 찾기 때문에 연산 효율 떨어진다.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def test(global_Actor, device, rank):
    env = gym.make("CartPole-v1")

    np.random.seed(seed+rank)
    random.seed(seed+rank)
    seed_torch(seed+rank)

    for e in range(episodes):

        total_reward = 0
        state = env.reset()[0]
        for stp in range(max_step):
            a_prob = global_Actor(torch.from_numpy(state).float().to(device))
            a_distribution = Categorical(a_prob)
            action = a_distribution.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            if stp == max_step-1:
                terminated = not terminated

            state = next_state
            total_reward += reward

        if e % 10 == 0:
            print("Episode : {} || TReward : {}".format(e, total_reward))

    env.close()

def train(global_Actor, global_Critic, device, rank):

    # Fixing seed for reducibility
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    seed_torch(seed + rank)

    env = gym.make("CartPole-v1")
    env_state_space  = env.observation_space.shape[0]
    env_action_space = env.action_space.n

    buffer = ReplayBuffer()

    local_Actor  = Actor(state_space=env_state_space, action_space=env_action_space, num_hidden_layer=hidden_layer_num, hidden_dim=hidden_dim_size).to(device)
    local_Critic = Critic(state_space=env_state_space, num_hidden_layer=hidden_layer_num, hidden_dim=hidden_dim_size).to(device)

    local_Actor.load_state_dict(global_Actor.state_dict())
    local_Critic.load_state_dict(global_Critic.state_dict())

    actor_optimizer  = optim.Adam(global_Actor.parameters(), lr = actor_lr)
    critic_optimizer = optim.Adam(global_Critic.parameters(), lr = critic_lr)

    batch = []
    for e in range(episodes):
        total_reward = 0
        terminated, truncated = False, False
        state = env.reset()[0]
        stp   = 0
        while (stp < max_step) or (terminated == False):
            policy = local_Actor(torch.from_numpy(state).float().to(device))
            a_disribution = Categorical(policy)
            action = a_disribution.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            batch.append([state, policy[action], reward, next_state, terminated, truncated])
            total_reward += reward

            if len(batch) > batch_size:
                states      = []
                next_states = []
                probs       = []
                rewards     = []
                terminateds = []
                truncateds  = []

                for item in batch:
                    states.append(item[0])
                    probs.append(item[1])
                    rewards.append(item[2])
                    next_states.append(item[3])
                    terminateds.append(item[4])
                    truncateds.append(item[5])

                states = torch.FloatTensor(states).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                terminateds = torch.FloatTensor(terminateds).to(device)
                truncateds = torch.FloatTensor(truncateds).to(device)

                v_s = local_Critic(states)
                v_prime = local_Critic(next_states)

                Q = rewards + discount_rate*v_prime.detach()*(1 - terminateds)
                A = Q - v_s

                critic_optimizer.zero_grad()
                critic_loss = F.mse_loss(v_s, Q.detach())
                critic_loss.backward()
                for global_param, local_param in zip(global_Critic.parameters(), local_Critic.parameters()):
                    global_param._grad = local_param.grad
                critic_optimizer.step()

                # Update Actor
                actor_optimizer.zero_grad()
                actor_loss = 0
                for idx, prob in enumerate(probs):
                    actor_loss += -A[idx].detach() * torch.log(prob)
                actor_loss /= len(probs)

                actor_loss.mean().backward()

                for global_param, local_param in zip(global_Actor.parameters(), local_Actor.parameters()):
                    global_param._grad = local_param.grad
                actor_optimizer.step()

                local_Actor.load_state_dict(global_Actor.state_dict())
                local_Critic.load_state_dict(global_Critic.state_dict())

                batch = []

            state = next_state
            stp += 1

        print("Thread : {} Episode : {} || TReward : {}".format(rank, e, total_reward))
    print(f"# ------------------------ P {rank} ends ----------------------------- #")
    env.close()

actor_lr = 1e-4
critic_lr = 1e-3
episodes = 10000
max_step = 20000
discount_rate = 0.99
batch_size = 5
seed = 777

hidden_layer_num = 2
hidden_dim_size = 128

if __name__ == "__main__":
    # set gym environment

    env = gym.make('CartPole-v1')
    env_state_space = env.observation_space.shape[0] # 2
    env_action_space = env.action_space.n            # 4

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Seed fixing
    #env.seed(seed)       # gym seed fix
    seed_torch(seed)     # torch seed fix
    random.seed(seed)    # random seed fix
    np.random.seed(seed) # numpy seed fix

    global_Critic = Critic(state_space=env_state_space, num_hidden_layer=hidden_layer_num, hidden_dim=hidden_dim_size).to(device)
    global_Actor = Actor(state_space=env_state_space, action_space=env_action_space, num_hidden_layer=hidden_layer_num,
                         hidden_dim=hidden_dim_size).to(device)

    env.close()

    global_Actor.share_memory()
    global_Critic.share_memory()

    processes   = []
    process_num = 5 # Number of threads

    """
        CUDA는 start method로 오직 spawn, forkserver만 제공한다.        
        -> multi-threaded 프로그램을 fork하는 순간 child process가 무조건 죽어버리는 현재의 os 디자인이 원인
    """
    mp.set_start_method('spawn')
    print("MP start method : ", mp.get_start_method())

    for rank in range(process_num):
        if rank == 0:
            p = mp.Process(target = test, args = (global_Actor, device, rank, ))
        else:
            p = mp.Process(target = train, args = (global_Actor, global_Critic, device, rank, ))
        p.start() # start threading
        processes.append(p)

    for p in processes:
        p.join()  # end threading

















