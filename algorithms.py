#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:45:39 2024

@author: chris
"""

from neural_net import CNN
from memories import ReplayMemory, RolloutMemory
import torch 
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
from torch.autograd import Variable
#from torchcontrib.optim import SWA

# Implementing Deep Q-Learning

class DQN:
    def __init__(self, input_shape, n_actions, gamma, initial_lr, initial_tau):
        self.gamma = gamma
        self.lr = initial_lr
        self.batch_size = 32
        self.input_shape = input_shape
        self.policy_net = CNN(self.batch_size, input_shape, n_actions)
        self.target_net = CNN(self.batch_size, input_shape, n_actions)
        self.memory = ReplayMemory(capacity = self.batch_size * 1000) 
        #self.optimizer = optim.Adam(params = self.policy_net.parameters(), lr=initial_lr)
        self.optimizer = optim.RMSprop(params = self.policy_net.parameters(), lr=initial_lr, eps=1e-5, alpha=0.99)
        self.tau = initial_tau
            
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(torch.load(filename))
        
    def get_next_state_qvalues(self, batch_next_states, game_over):
        
        #print(f"batch_next_states: {batch_next_states}.\nGame over: {game_over}")

        non_final_next_states = torch.cat([s for (s, f) in zip(batch_next_states, game_over)
                                           if f is False])

        #print(f"non_final_next_states: {non_final_next_states}.")
        
        next_states_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            qvalues = self.target_net(non_final_next_states)
            #print(f"qvalues: {qvalues}.")
            #print(f"qvalues max: {qvalues.max(1)[0]}.")
            mask = ~torch.tensor(game_over)
            #print(f"mask: {mask}")
            next_states_values[mask] = qvalues.max(1)[0]
            
        #print(f"next_states_values: {next_states_values}.")
        return next_states_values.detach() #.squeeze(-1)

    
    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states, game_over):
        '''
        This function compute the temporal difference, and accordingly the loss,
        and update the weights with our optimizer in order to reduce that loss.
        parameters:
            batch_states: A batch of input states.
            batch_actions: A batch of actions played.
            batch_rewards: A batch of the rewards received.
            batch_next_states: A batch of the next states reached.
            game_over: A batch of booleans that indicates is the game ended
        '''
        
        #print(f"batch_states: {batch_states}.")
        #print(f"batch_actions: {batch_actions}.")
        
        batch_states = torch.cat(batch_states)
        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.cat(batch_rewards)

        #print(f"batch_states: {batch_states}. shape: {batch_states.shape}")
        #print(f"batch_actions: {batch_actions}. shape: {batch_actions.shape}")
        #print(f"batch_rewards: {batch_rewards}. shape: {batch_rewards.shape}")
        
        # State-action values for the batch
        batch_outputs = self.policy_net(batch_states)
        #print(f"batch_outputs: {batch_outputs}. shape: {batch_outputs.shape}")

        # Selection of each output corresponding to the actions indexes
        batch_outputs = batch_outputs.gather(1, batch_actions)
        #print(f"selected batch_outputs: {batch_outputs}. shape: {batch_outputs.shape}")
        
        next_states_values = self.get_next_state_qvalues(batch_next_states, game_over)

        # Compute the expected state action values
        batch_targets = batch_rewards + self.gamma * next_states_values
        #print(f"batch_targets = {batch_targets}. shape: {batch_targets.shape}")
        
        # Update the model with gradiant descent
        self.optimizer.zero_grad() #  Put back gradient to 0
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets.unsqueeze(1))
        td_loss.backward()
        # gradient-clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.6)
        #torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update the target net
        # θ′ ← τ θ + (1 − τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        #self.tau = 0.05
        self.tau = 0.1

    def update(self):
        if len(self.memory.memory) > 10*self.batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, game_over = self.memory.sample(self.batch_size)
            #print(f"{batch_states, batch_actions, batch_rewards, batch_next_states, game_over}")
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states, game_over)




# Implementing Advantage Actor-Critic

class A2C:
    def __init__(self, input_shape, n_actions, n_steps, n_envs, gamma, initial_lr):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.entropy_coef = 0.03 #0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gamma = gamma
        self.lr = initial_lr
        self.input_shape = input_shape
        
        self.a2c_net = CNN(1, input_shape, n_actions, 1)

        self.memory = [RolloutMemory(n_steps, input_shape) for _ in range(n_envs)]
        #self.optimizer = optim.Adam(params = self.a2c_net.parameters(), lr=initial_lr)
        self.optimizer = optim.RMSprop(params = self.a2c_net.parameters(), lr=initial_lr, eps=1e-5, alpha=0.99)
        #self.optimizer = optim.SGD(self.a2c_net.parameters(), lr=initial_lr)
        
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
        
        
    def save(self, filename):
        torch.save(self.a2c_net.state_dict(), filename)


    def load(self, filename):
        self.a2c_net.load_state_dict(torch.load(filename))
    
    
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
                _, next_value_pred = self.a2c_net(Variable(memory.get_last_state())) #.view(-1, *self.input_shape))
                next_value_pred = next_value_pred.data

        returns = memory.calc_nsteps_returns(next_value_pred, self.gamma)

        # Get state-action pairs
        states, actions = memory.get_s_a()
        
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
        