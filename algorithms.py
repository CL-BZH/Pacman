#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:45:39 2024

@author: chris
"""

from neural_net import CNN, DuelingCNN, get_device
from memories import ReplayMemory, RolloutMemory, NStepReplayBuffer
import torch 
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
from torch.autograd import Variable
#from torchcontrib.optim import SWA

# Implementing Deep Q-Learning

class DQN:
    '''
    Deep Q-Network with the improvements that matter most in practice:
      - Double DQN targets (van Hasselt et al., 2016): the online network picks
        the next action, the target network evaluates it. Removes the
        systematic over-estimation of the max operator.
      - Dueling architecture (see neural_net.DuelingCNN).
      - n-step returns (default n=3): rewards propagate n times faster, which
        helps a lot with the sparse "win"/"eat ghost" rewards.
      - Polyak-averaged target network with a small tau (default 0.005),
        i.e. a target that moves slowly enough to stabilise learning.
      - Reward scaling (default x0.1) so that TD errors stay in the range
        where the Huber loss and Adam behave well.
      - Everything runs on GPU when available.
    '''
    def __init__(self, input_shape, n_actions, gamma=0.99, initial_lr=2.5e-4, initial_tau=0.005,
                 batch_size=128, buffer_capacity=200_000, n_envs=1, n_step=3,
                 double=True, network="dueling", reward_scale=0.1, max_grad_norm=10.,
                 device=None):
        self.device = get_device(device)
        self.gamma = gamma
        self.lr = initial_lr
        self.tau = initial_tau
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.double = double
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        self._build(network)
        self.memory = NStepReplayBuffer(buffer_capacity, input_shape, n_envs,
                                        n_step=n_step, gamma=gamma, device=self.device)

    def _build(self, network):
        self.network = network
        if network == "dueling":
            make = lambda: DuelingCNN(self.input_shape, self.n_actions)
        elif network == "legacy":
            # Original architecture (to load old checkpoints)
            make = lambda: CNN(self.batch_size, self.input_shape, self.n_actions)
        else:
            raise ValueError(f"Unknown network '{network}'")
        self.policy_net = make().to(self.device)
        self.target_net = make().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, eps=1.5e-4)

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=self.device)
        # Checkpoints made with the original CNN have 'fc2' as output layer
        network = "legacy" if "fc2.weight" in state_dict else "dueling"
        if network != self.network:
            self._build(network)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def qvalues(self, states):
        '''Q-values of a batch of states (numpy array or tensor).'''
        if not torch.is_tensor(states):
            states = torch.from_numpy(states)
        return self.policy_net(states.to(self.device, dtype=torch.float32))

    def learn(self):
        '''
        One gradient step on a mini-batch sampled from the replay memory.
        Returns (loss, mean predicted Q) as detached tensors for monitoring
        (no .item() here, so the GPU is not synchronised at every step).
        '''
        states, actions, returns, next_states, dones, discounts = self.memory.sample(self.batch_size)

        q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_target = self.target_net(next_states)
            if self.double:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q = next_q_target.max(dim=1)[0]
            targets = returns * self.reward_scale + discounts * (1. - dones) * next_q

        loss = F.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Polyak update of the target net: θ′ ← τ θ + (1 − τ) θ′
        with torch.no_grad():
            for p_t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
                p_t.lerp_(p, self.tau)

        return loss.detach(), q.mean().detach()




# Implementing Advantage Actor-Critic

class A2C:
    def __init__(self, input_shape, n_actions, n_steps, n_envs, gamma, initial_lr, device=None):
        self.device = get_device(device)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.entropy_coef = 0.03 #0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gamma = gamma
        self.lr = initial_lr
        self.input_shape = input_shape
        
        self.a2c_net = CNN(1, input_shape, n_actions, 1).to(self.device)

        self.memory = [RolloutMemory(n_steps, input_shape) for _ in range(n_envs)]
        #self.optimizer = optim.Adam(params = self.a2c_net.parameters(), lr=initial_lr)
        self.optimizer = optim.RMSprop(params = self.a2c_net.parameters(), lr=initial_lr, eps=1e-5, alpha=0.99)
        #self.optimizer = optim.SGD(self.a2c_net.parameters(), lr=initial_lr)
        
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
        
        
    def save(self, filename):
        torch.save(self.a2c_net.state_dict(), filename)


    def load(self, filename):
        self.a2c_net.load_state_dict(torch.load(filename, map_location=self.device))
    
    
    def evaluate_actions(self, state, action):
        logit, value = self.a2c_net(state)
        
        probs     = F.softmax(logit, dim=-1)
        log_probs = F.log_softmax(logit, dim=-1)
        
        action_log_probs = log_probs.gather(1, action)
        #print(f"state: {state}.\naction: {action}.\nlog_probs: {log_probs}.\naction_log_probs: {action_log_probs}")
        entropy = -(probs * log_probs).sum(1).mean()
        
        return logit, action_log_probs, value, entropy


    def compute_losses(self, env_idx, n_steps, end):
        '''
        
        '''
        #print(f"last_step_idx: {last_step_idx}")

        memory = self.memory[env_idx]
        
        # Predict the value of the last state
        next_value_pred = 0
        if end == False:
            with torch.no_grad():
                _, next_value_pred = self.a2c_net(memory.get_last_state().to(self.device))
                next_value_pred = next_value_pred.cpu()

        returns = memory.calc_nsteps_returns(next_value_pred, self.gamma).to(self.device)

        # Get state-action pairs
        states, actions = memory.get_s_a()
        states, actions = states.to(self.device), actions.to(self.device)
        
        logit, action_log_probs, values, entropy = self.evaluate_actions(states, actions)

        values = values.view(n_steps, 1)
        action_log_probs = action_log_probs.view(n_steps, 1)

        with torch.no_grad():
            advantages = returns - values
        
        value_loss = advantages.pow(2).mean()
        policy_loss = -(Variable(advantages.data) * action_log_probs).mean()
        policy_loss -= entropy * self.entropy_coef

        return value_loss, policy_loss

    
    def update(self, value_losses, policy_losses):
        '''
        
        '''
        #print(f"value_losses: {value_losses}. policy_losses: {policy_losses}")
        #r = len(value_losses)
        value_losses = torch.stack(value_losses) #.view((r,-1))
        policy_losses = torch.stack(policy_losses) #.view((r,-1))
        
        # Update the model with gradient descent
        self.optimizer.zero_grad() #  Put back gradient to 0
        loss = value_losses.mean() * self.value_loss_coef + policy_losses.mean()
        loss.backward()
        # gradient-clipping
        #torch.nn.utils.clip_grad_value_(self.a2c_net.parameters(), 100)
        nn.utils.clip_grad_norm_(self.a2c_net.parameters(), self.max_grad_norm)
        self.optimizer.step()


    def roll_memory(self, env_idx):
        # Last state becomes the first one
        # (if it is an end state it will be erased during the reset_episode)        
        self.memory[env_idx].roll()


    def reset_memory(self, env_idx, state=None):
        self.memory[env_idx].reset(state)
        