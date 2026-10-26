#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:01:30 2024

@author: chris
"""

import numpy as np
from collections import deque
import torch 


class NStepReplayBuffer:
    '''
    Pre-allocated circular replay buffer that stores n-step transitions
        (s_t, a_t, R_t, s_{t+m}, done, gamma^m)
    with R_t = r_t + gamma r_{t+1} + ... + gamma^{m-1} r_{t+m-1} and m <= n_step.

    Transitions are pushed one step at a time for each environment; the n-step
    returns are assembled internally. When an episode ends (terminal or
    truncated) the pending transitions are flushed with shorter horizons.
    'done' is True only for real terminal states (Pacman eaten or game won):
    for truncated (aborted) games the target still bootstraps from s_{t+m}.

    States are stored in float16 on the CPU (values are in [0, 1]) to halve
    memory, and converted to float32 on the training device when sampled.
    '''
    def __init__(self, capacity, state_shape, n_envs, n_step=3, gamma=0.99, device="cpu"):
        self.capacity = int(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.float16)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.float16)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.pending = [deque() for _ in range(n_envs)]

    def __len__(self):
        return self.size

    def _store(self, s, a, ret, s_next, done, discount):
        i = self.idx
        self.states[i] = s
        self.actions[i] = a
        self.returns[i] = ret
        self.next_states[i] = s_next
        self.dones[i] = done
        self.discounts[i] = discount
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _emit_first(self, env_idx, s_next, done):
        q = self.pending[env_idx]
        ret = 0.
        for k, (_, _, r) in enumerate(q):
            ret += (self.gamma ** k) * r
        s, a, _ = q.popleft()
        self._store(s, a, ret, s_next, done, self.gamma ** (len(q) + 1))

    def push(self, env_idx, state, action, reward, next_state, done, truncated=False):
        q = self.pending[env_idx]
        q.append((state, action, reward))
        if done or truncated:
            while q:
                self._emit_first(env_idx, next_state, float(done))
        elif len(q) == self.n_step:
            self._emit_first(env_idx, next_state, 0.)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x, dt=torch.float32: torch.from_numpy(x[idx]).to(self.device, dtype=dt, non_blocking=True)
        return (to(self.states), to(self.actions, torch.int64), to(self.returns),
                to(self.next_states), to(self.dones), to(self.discounts))
