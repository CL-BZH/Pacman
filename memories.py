#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:01:30 2024

@author: chris
"""

import random 
import numpy as np
from collections import deque
import torch 
from torch.autograd import Variable


class ReplayMemory:
    def __init__(self, capacity): 
        self.capacity = capacity 
        self.memory = [] 

    def push(self, event):
        #print(f"Push to memeory: {event}")
        self.memory.append(event)
        if len(self.memory) > self.capacity: 
            del self.memory[0]

    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return samples

        

class RolloutMemory:
    def __init__(self, n_steps, state_shape):
        #print(f"Init rollout memory. n steps: {n_steps}. state shape: {state_shape}")
        self.n_steps = n_steps
        self.state_shape = state_shape
        self.insert_idx = 0
        self.reset()
        
    def reset(self, state=None):
        self.insert_idx = 0
        self.states  = torch.zeros(self.n_steps + 1, 1, *self.state_shape)
        if state != None:
            self.states[0].copy_(state)  
        self.rewards = torch.zeros(self.n_steps, 1).type(torch.float32) 
        self.actions = torch.zeros(self.n_steps, 1).type(torch.int64)
        
    def insert(self, state, action, reward):
        #print(f"Insert state of shape: {state.shape}")
        #assert self.insert_idx < self.n_steps
        self.states[self.insert_idx + 1].copy_(state)
        self.actions[self.insert_idx] = action
        self.rewards[self.insert_idx] = reward
        self.insert_idx += 1

    def get_last_state(self):
        return self.states[self.insert_idx]

    def get_last_reward(self):
        return self.rewards[self.insert_idx-1]
        
    def roll(self):
        """
        Move the last inserted state to the beginning.
        All other fields of memory are reset.
        """
        self.reset(self.get_last_state())
        
    def calc_nsteps_returns(self, next_value_pred, gamma):
        # Last state reached is stored at index 'self.insert_idx'
        returns_size = self.insert_idx + 1
        returns = torch.zeros(returns_size, 1)
        returns[-1] = next_value_pred
        for step in reversed(range(self.insert_idx)):
            #print(f"step: {step}. returns[step+1] = {returns[step + 1]}. rewards[step] = {self.rewards[step]}")
            returns[step] = self.rewards[step] + gamma * returns[step + 1]
        return Variable(returns[:-1])

    def get_s_a(self):
        # Get all action but last
        s = Variable(self.states[:self.insert_idx]).view(-1, *self.state_shape)
        # Get actions that were taken in each state stored in 's'
        a = Variable(self.actions[:self.insert_idx]).view(-1, 1)
        return s, a
    


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
