#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:45:39 2024

@author: chris
"""

from neural_net import DuelingCNN, ActorCriticCNN, get_device
from memories import NStepReplayBuffer
import numpy as np
import torch 
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  

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
        if network != "dueling":
            raise ValueError(f"Unknown network '{network}'")
        self.network = network
        make = lambda: DuelingCNN(self.input_shape, self.n_actions)
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




# Implementing Advantage Actor-Critic (with PPO updates)

class A2C:
    '''
    Actor-critic trained on rollouts of n_steps x n_envs transitions.

    - Advantages: GAE(lambda) (Schulman et al., 2016), normalised per mini-batch.
    - Updates: PPO clipped objective (Schulman et al., 2017) with 'ppo_epochs'
      passes over the rollout split into 'n_minibatches' mini-batches.
      With ppo_epochs=1 and n_minibatches=1 the probability ratio is always 1,
      the clipping never acts, and the update is exactly A2C with GAE.
    - Rewards are scaled by 'reward_scale' so values stay of order 1.
    '''
    def __init__(self, input_shape, n_actions, gamma=0.99, initial_lr=2.5e-4, gae_lambda=0.95,
                 entropy_coef=0.03, value_loss_coef=0.5, max_grad_norm=0.5,
                 ppo_epochs=4, n_minibatches=4, clip_eps=0.2, reward_scale=0.1,
                 action_mirror=None, device=None):
        self.device = get_device(device)
        self.gamma = gamma
        self.lr = initial_lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.n_minibatches = n_minibatches
        self.clip_eps = clip_eps
        self.reward_scale = reward_scale
        self.input_shape = input_shape
        self.a2c_net = ActorCriticCNN(input_shape, n_actions, action_mirror=action_mirror).to(self.device)
        self.optimizer = optim.Adam(self.a2c_net.parameters(), lr=initial_lr, eps=1e-5)

    def save(self, filename):
        torch.save(self.a2c_net.state_dict(), filename)

    def load(self, filename):
        self.a2c_net.load_state_dict(torch.load(filename, map_location=self.device))

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _tensor(self, states):
        if not torch.is_tensor(states):
            states = torch.from_numpy(states)
        return states.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def act(self, states, stochastic=True):
        '''
        Actions for a batch of states, with their log-probabilities and the
        state values (all as numpy arrays).
        '''
        logits, values = self.a2c_net(self._tensor(states))
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample() if stochastic else logits.argmax(dim=1)
        return (actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy(), values.cpu().numpy())

    @torch.no_grad()
    def values(self, states):
        return self.a2c_net(self._tensor(states))[1].cpu().numpy()

    def compute_gae(self, rewards, values, dones, last_values):
        '''
        rewards, values, dones: arrays [n_steps, n_envs]; last_values: [n_envs]
        (value of the states reached after the last step).
        dones[t] is 1 if the game ended (terminal state) after the action at step t.
        Returns (advantages, returns), both [n_steps, n_envs].
        '''
        n_steps = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        last_gae = 0.
        for t in reversed(range(n_steps)):
            next_values = last_values if t == n_steps - 1 else values[t + 1]
            not_done = 1. - dones[t]
            delta = rewards[t] + self.gamma * next_values * not_done - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * not_done * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values

    def update(self, states, actions, old_log_probs, advantages, returns):
        '''
        PPO / A2C update on a rollout (all inputs flattened over steps and envs).
        Returns a dict of monitoring statistics.
        '''
        states = self._tensor(states)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        old_log_probs = torch.as_tensor(old_log_probs, device=self.device, dtype=torch.float32)
        advantages = torch.as_tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(returns, device=self.device, dtype=torch.float32)

        batch_size = states.shape[0]
        mb_size = batch_size // self.n_minibatches
        stats = {"policy_loss": 0., "value_loss": 0., "entropy": 0., "approx_kl": 0., "clip_frac": 0.}
        n = 0
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for k in range(self.n_minibatches):
                idx = perm[k * mb_size:(k + 1) * mb_size]
                logits, values = self.a2c_net(states[idx])
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = log_probs - old_log_probs[idx]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(ratio * adv,
                                         ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv).mean()
                value_loss = 0.5 * (returns[idx] - values).pow(2).mean()
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.a2c_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                n += 1
        return {k: v / n for k, v in stats.items()}
