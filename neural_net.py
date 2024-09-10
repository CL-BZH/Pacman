#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:49:52 2024

@author: chris
"""

import torch 
import torch.nn as nn  
from torch.autograd import Variable

# Creating the architecture of the Convolutional Neural Network

class CNN(nn.Module):
    def __init__(self, batch_size, input_shape, n_actions, sup=0):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_actions = n_actions # Size of the output layer for the actions
        self.sup = sup # Supplementary output layer
        #print(f"CNN: input shape {self.input_shape}, output size: {self.n_actions}")
        
        in_channels=self.input_shape[0]
        # Featurizing layers
        self.features = nn.Sequential( 
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.Conv2d(16, 32, kernel_size=1, stride=1),
            #nn.ReLU(),
        )
        self.fc1_in_size = self._feature_size()
        
        # Fully connected hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(self.fc1_in_size, 256),
            nn.ReLU(),
        )
        # Fully connected output layer for the actions
        self.fc2 = nn.Linear(256, self.n_actions)

        # If sup > 0, then add another output layer
        if self.sup > 0:
            self.fc3 = nn.Linear(256, self.sup)

    def _feature_size(self):
        # For pytorch the number of channel comes first [B, C, H, W]
        #(TensorFlow support channel last order [B, H, W, C])
        x = Variable(torch.zeros(1, *self.input_shape))
        #print(f"input x shape: {x.shape}")
        x = self.features(x)
        #print(f"conv out x shape: {x.shape}")
        size = x.view(1, -1).size(1)
        #print(f"x shape: {x.shape} size: {size}")
        return size

    # forward() is called in __call__ of nn.Module.
    def forward(self, state): 
        '''
        Forward-propagates the signal inside the neural network
        and returns the predicted Q-values.
        '''
        #print(f"CNN.forward({state.shape})")
        x = self.features(state)
        x = x.view(x.size(0), -1) # flattening the last convolutional layer into this 1D vector x
        x = self.fc1(x)
        if self.sup > 0: 
            return self.fc2(x), self.fc3(x)
        return self.fc2(x)
