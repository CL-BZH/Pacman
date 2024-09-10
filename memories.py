#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:01:30 2024

@author: chris
"""

import random 
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
    
