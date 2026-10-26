#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:49:52 2024

@author: chris
"""

import torch
import torch.nn as nn


def get_device(device=None):
    '''
    Return the torch device to use: CUDA if available, then Apple MPS, else CPU.
    A device (or its name) can be forced with the 'device' argument.
    '''
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DuelingCNN(nn.Module):
    '''
    Q-network for DQN.
    The board is small (9 x 13 once borders are removed), so the convolutions
    keep the full resolution (stride 1, padding 1): a stride-2 convolution
    blurs the exact cell positions of Pacman, the ghosts and the cookies,
    which is the information the agent needs most.
    The output uses the dueling decomposition Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    (Wang et al., 2016), which learns state values faster when many actions
    have similar values (very common in Pacman corridors).
    '''
    def __init__(self, input_shape, n_actions, channels=(32, 64, 32), hidden=256):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        in_channels = input_shape[0]
        layers = []
        for c in channels:
            layers += [nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1), nn.ReLU()]
            in_channels = c
        self.features = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, *input_shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, state):
        x = self.fc(self.features(state))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


class ActorCriticCNN(nn.Module):
    '''
    Actor-critic network for A2C/PPO: a full-resolution convolutional trunk
    shared by a policy head (action logits) and a value head.

    If 'action_mirror' is given (permutation of the actions under a left-right
    mirror of the board, e.g. East <-> West), the network is made exactly
    equivariant to that symmetry: the outputs for the state and for its mirror
    image are averaged,
        logits(s) = (f(s) + P f(mirror(s))) / 2,  V(s) = (V(s) + V(mirror(s))) / 2.
    The agent then cannot learn a different behaviour on the two halves of a
    symmetric board, which is equivalent to doubling the training data.
    '''
    def __init__(self, input_shape, n_actions, channels=(32, 64, 32), hidden=256, action_mirror=None):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        in_channels = input_shape[0]
        layers = []
        for c in channels:
            layers += [nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1), nn.ReLU()]
            in_channels = c
        self.features = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, *input_shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)
        # Small initial policy weights: the initial policy is close to uniform
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        if action_mirror is not None:
            self.register_buffer("action_mirror", torch.as_tensor(action_mirror, dtype=torch.long))
        else:
            self.action_mirror = None

    def _forward(self, state):
        x = self.fc(self.features(state))
        return self.policy(x), self.value(x).squeeze(-1)

    def forward(self, state):
        '''Returns (logits [B, n_actions], values [B]).'''
        if self.action_mirror is None:
            return self._forward(state)
        logits, values = self._forward(torch.cat([state, torch.flip(state, dims=[-1])]))
        b = state.shape[0]
        logits = (logits[:b] + logits[b:][:, self.action_mirror]) / 2
        values = (values[:b] + values[b:]) / 2
        return logits, values
