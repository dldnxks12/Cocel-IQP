"""

* reference

            : https://mrsyee.github.io/rl/2019/01/25/PER-sumtree/
            : https://github.com/thomashirtz/q-learning-algorithms/blob/master/prioritized_experience_replay/per.py
            : Build data structure for storing prioritized experiences
            : https://yiyj1030.tistory.com/491

"""

import numpy as np
from itertools import tee


def pairwise(iterable):  # "ABCDE" -> AB BC CD DE ..
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class SumTree:
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity  # memory capacity of replay buffer ( N )
        self.data = np.zeros(capacity, dtype=object)  # Leaf level data (Actual TD Errors)
        self.tree = np.zeros(2 * capacity - 1)  # TD error 배열의 2배의 메모리 공간이 필요

    # Update tree - ok
    def _update_tree(self, idx, value):
        difference = value - self.tree[idx]  # 기존 값과 새로 업데이트할 값과의 차이
        self.tree[idx] = value  # 새로운 값으로 해당 값 변경

        while idx > 0:  # propagate this change to the root node
            idx = (idx - 1) // 2  # parent node
            self.tree[idx] += difference  # 변화 반영

    # Update tree - ok
    def update(self, indexes, values):  # index -> td error에 대한 index
        for index, value in zip(indexes, values):
            self._update_tree(index, value)

    # ok
    def push(self, value, data):  # value : priority, data = experience (transition)
        tree_idx = self.data_pointer + self.capacity - 1  # leaf node부터 시작해서 업데이트 해야하기 때문

        # data 배열은 leaf level을 다룸
        self.data[self.data_pointer] = data
        self._update_tree(tree_idx, value)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def get(self, value):  # t_summation을 일정하게 나눈 구간인 각 Segment 들에서 랜덤하게 값을 추출 : value
        idx = self._retrieve(0, value)  # root 부터 시작해서 leaf value 찾기
        idx_data = idx - self.capacity + 1  # leaf level node들의 index
        return idx, self.tree[idx], self.data[idx_data]

    # Sampling !!  - 실제 데이터는 맨 아래 있는 leaf node 들이 가지고 있다.  # ok
    # Find Sample on leaf node
    def _retrieve(self, idx, value):  # root idx = 0 으로 설정
        left_idx = 2 * idx + 1  # 왼쪽 노드 index
        right_idx = 2 * idx + 2  # 오른쪽 노드 index

        if left_idx >= len(self.tree):
            return idx

        if value <= self.tree[left_idx]:
            return self._retrieve(left_idx, value)
        else:
            return self._retrieve(right_idx, value - self.tree[left_idx])

    @property  # ok
    def total(self):
        return self.tree[0]  # root node -> total summation of TD errors

    @property  # ok
    def max_leaf_value(self):
        return np.max(self.tree[-self.capacity:])  # leaf node 중에서 젤 큰놈 return

    @property  # ok
    def min_leaf_value(self):
        # leaf node 중에서 0이 아닌 젤 작은 놈 return
        return np.min(self.tree[-self.capacity:][np.nonzero(self.tree[-self.capacity:])])

    def __len__(self):  # ok
        # 0이 아닌 leaf들의 개수 -> 길이
        return np.count_nonzero(self.tree[-self.capacity:])


class ProportionalPrioritizedMemory:
    def __init__(self, capacity, epsilon=0.01):
        self.epsilon = epsilon
        self.capacity = capacity
        self.maximum_priority = 1.0
        self.tree = SumTree(capacity)

    def push(self, experience):  # memory에 transition 넣기
        # 모든 experience가 적어도 1번은 뽑힐 수 있도록 TD error가 계산되지 않은 친구는 최대 확률로 넣어준다.
        priority = self.tree.max_leaf_value if self.tree.max_leaf_value else self.maximum_priority
        self.tree.push(value = priority, data=experience)

    def sample(self, batch_size, beta=0.4):  # beta : importance sampling weight
        segments = np.linspace(0, self.tree.total, num=batch_size + 1)  # total summation을 균등하게 쪼개기

        tuples = []
        # 각 Segment 마다 sampling 수행 -> proportional prioritization 수행
        for start, end in pairwise(segments):
            value = np.random.uniform(start, end)
            tuples.append(self.tree.get(value))  # value와 가장 비슷한 값 찾아서 return 하는 것 같다. return 되는 건 실제 leaf value !
        indexes, priorities, experiences = zip(*tuples)  # Unpack

        probabilities = np.array(priorities) / self.tree.total
        maximum_weight = (self.tree.total / self.capacity * self.tree.min_leaf_value) ** beta  # this is for stablizing
        weights = ((1 / self.capacity * probabilities) ** beta) / maximum_weight

        return list(indexes), list(weights), list(experiences)

    def update(self, indexes, deltas, alpha=0.6):
        priorities = (np.abs(deltas) + self.epsilon) ** alpha
        clipped_priorities = np.clip(priorities, 0, self.maximum_priority)
        self.tree.update(indexes, clipped_priorities)

    def __len__(self):
        return len(self.tree)
