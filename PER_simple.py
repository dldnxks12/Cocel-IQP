"""

*reference : https://github.com/rlcode/per/blob/master/cartpole_per.py

"""

import random
import numpy as np

class SumTree:
    write = 0 # data pointer

    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1)
        self.data      = np.zeros(capacity, dtype = object)
        self.n_entries = 0

    def _propagate(self, idx, difference):
        parent = (idx - 1) // 2 # Upper node (Parent node)

        self.tree[parent] += difference

        if parent != 0:
            self._propagate(parent, difference) # propagate further

    def _retrieve(self, idx, sample):   # Leaf level에 달린 TD Error 값들
        left = 2 * idx + 1 # left child node
        right =2 * idx + 2 # right child node

        if left >= len(self.tree):
            return idx

        if sample <= self.tree[left]:
            return self._retrieve(left, sample)
        else:
            return self._retrieve(right, sample - self.tree[left])

    def total(self):
        return self.tree[0]  # total summation of tree is at the root node

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p): # update priority
        difference = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, difference)

    def get(self, sample): # get priority and sample
        idx = self._retrieve(0, sample)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    @property  # ok
    def max_leaf_value(self):
        return np.max(self.tree[-self.capacity:])  # leaf node 중에서 젤 큰놈 return

    def __len__(self):  # ok
        # 0이 아닌 leaf들의 개수 -> 길이
        return np.count_nonzero(self.tree[-self.capacity:])

class Memory:
    e     = 0.01
    alpha = 0.6
    beta  = 0.4
    beta_increment_per_sampling = 0.001


    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.maximum_priority = 1.0

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def add(self, sample):
        p = self.tree.max_leaf_value if self.tree.max_leaf_value else self.maximum_priority
        self.tree.add(p, sample)

    def sample(self, n):
        batch      = []
        idxs       = []
        segment    = self.tree.total() / n
        priorities = []

        self.beta = np.min([1. , self.beta + self.beta_increment_per_sampling]) # maxmimum 1

        for i in range(n):
            a = segment * i       # i번 째 segment
            b = segment * (i + 1) # i + 1 번째 segment

            s = random.uniform(a, b) # 이 구간 사이에서 uniform sampling
            (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return idxs, is_weight, batch

    def update(self, idxes, errors):
        for idx, error in zip(idxes, errors):
            p = self._get_priority(error)
            clipped_p = np.clip(p, 0, self.maximum_priority)
            self.tree.update(idx, clipped_p)

    def __len__(self):
        return len(self.tree)
