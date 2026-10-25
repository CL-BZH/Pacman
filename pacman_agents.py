#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:38:31 2024

@author: chris
"""

from pacman_env import PacmanEnvironment
from ghost_agents import EdibleGhostBehaviors
from algorithms import DQN, A2C
import random
import numpy as np
from scipy.spatial.distance import cdist
import time
from datetime import timedelta
import numpy.matlib
from abc import ABC, abstractmethod
import torch 
import torch.nn.functional as F  
from typing import List

#from matplotlib import pyplot as plt
#from IPython.display import display
import pandas as pd
import pathlib
from collections import deque

class PacmanAgent(ABC):
    def __init__(self, pacman_envs: List[PacmanEnvironment], n_frames, grayscale):
        self.gamma = 1
        self.episode_total_reward = 0
        self.pacman_envs = pacman_envs
        self.n_envs = len(pacman_envs)
        self.weights = np.ones(len(self.pacman_envs[0].valid_pos))
        self.reward_per_episode = []
        self.actions = self.pacman_envs[0].get_all_actions()
        self.n_actions = len(self.actions)
        self.current_state = []
        self.episode_history = []
        self.episode_rewards = []
        self.total_reward_per_episode = []

        rows, columns, channels = self.pacman_envs[0].rgb_image_shape
        self.img_n_channels = channels # Channels of an image
        self.n_frames = n_frames # number of frames (images) per state
        # Store image frames
        self.image_frames = np.zeros((self.img_n_channels * self.n_frames, rows, columns), dtype=np.float32)
        
        self.cnn_in_channels = self.img_n_channels * self.n_frames # Input size of the CNN
        if grayscale == True:
            self.img_n_channels = 1 # grayscale
            self.cnn_in_channels = self.n_frames
            self.input_shape = (self.cnn_in_channels, rows, columns)
        else:
            self.input_shape = (self.cnn_in_channels, rows, columns)
        
        
    def reset(self):
        for i in range(self.n_envs):
            self.pacman_envs[i].reset()
        self.episode_total_reward = 0
        self.reward_per_episode = []
        self.current_state = []
        self.episode_history = []
        self.episode_rewards = []
        self.total_reward_per_episode = []
        
    def reset_episode(self):
        self.current_state = []
        for env_idx in range(self.n_envs):
            #print(f"Reset episode for env {env_idx}")
            
            self.episode_total_reward = 0
            
            self.reset_env(env_idx)
    
            self.episode_history = []
            self.episode_rewards = []
            
            # Child class specific reset
            self.child_reset_episode(env_idx)
            
        
    def reset_env(self, env_idx):
        '''
        Reset one environment with a start position chosen so that
        rarely used start positions are favoured.
        '''
        start_pos = None
        while start_pos == None:
            pos_selection_distribution = np.reciprocal(self.weights)
            pos_selection_distribution = pos_selection_distribution / np.sum(pos_selection_distribution)
            idx  = np.random.choice(list(range(len(self.pacman_envs[env_idx].valid_pos))), p=pos_selection_distribution)
            
            start_pos = self.pacman_envs[env_idx].valid_pos[idx]
            self.weights[idx] += 1
            
            if (self.pacman_envs[env_idx].start_ghosts_pos != None and
                np.min(cdist([start_pos], self.pacman_envs[env_idx].start_ghosts_pos)) < 2):
                start_pos = None
        
        self.pacman_envs[env_idx].reset(start_pos)

    def observation(self, env_idx):
        '''
        Current image of an environment as a (channels, height, width) float32 array.
        '''
        img = self.image_processing(self.pacman_envs[env_idx].rgb_image)
        if self.img_n_channels > 1:
            img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(img, dtype=np.float32)

    def set_gamma(self, gamma):
        self.gamma = gamma

    def image_processing(self, img):
            
        def rgb2gray(rgb):
            # Check for example OpenCV
            scale = np.array([0.2989, 0.5870, 0.1140], dtype='float32')
            img = np.dot(rgb[...,:3], scale)
            return np.expand_dims(img, axis=0) #img[np.newaxis,...]
            
        if self.img_n_channels == 1:
            # transform to grayscale
            img = rgb2gray(img)
            
        return img

    def frames_to_state(self):
        '''
        Take the stored image frames and return a state.
        '''
        state = self.image_frames
        # nn.Conv2d input: (1 x channels x height x width) 
        state = state.view(-1, *state.size()[2:])
        return state.unsqueeze(0)
        
    def image_to_state(self, np_img):
        '''
        Take a numpy image and reshape it into a tensor that
        can pass through nn.Conv2d.
        '''
        tensor_image = torch.from_numpy(np_img)
        # (height, width, channels) -> (channels, height, width)
        if self.img_n_channels > 1:
            tensor_image = tensor_image.permute(2, 0, 1)

        self.episode_history.append(tensor_image)
        
        # Replace the oldest frame with the new one
        stack = []
        for i in range(self.n_frames - 1):
            stack.append(self.image_frames[i+1, :, :])
        stack.append(tensor_image)
        self.image_frames = torch.stack(stack)
        
        return self.frames_to_state()
        
    @abstractmethod
    def child_reset_episode(self, env_idx):
        pass
        
    @abstractmethod
    def epsilon_greedy(self, s, eps):
        '''
        Epsilon-greedy policy
        '''
        pass

    @abstractmethod
    def take_action(self, **kwarg):
        '''
        Take an action according to a given policy.
        '''
        raise NotImplementedError("Must override take_action")

    @abstractmethod
    def run_episodes(self, episode_max_length=500, total_episodes=1000):
        pass     
    


class DQN_PacmanAgent(PacmanAgent):
    '''
    DQN agent.

    Training (run_episodes) steps all the environments in lockstep: at each
    step the Q-network evaluates the states of all environments in a single
    batched forward pass (efficient on GPU), every environment performs one
    move, finished games are restarted immediately, and the network is updated
    every 'train_freq' steps. Schedules (epsilon, learning rate) are expressed
    in environment steps rather than in episodes.
    '''
    def __init__(self, pacman_envs: List[PacmanEnvironment], gamma=0.99, initial_lr=2.5e-4, initial_tau=0.005,
                 grayscale=False, n_frames=1, batch_size=128, buffer_capacity=200_000, n_step=3,
                 double=True, network="dueling", reward_scale=0.1, device=None):
        super().__init__(pacman_envs, n_frames, grayscale)
        self.epsilon = 1
        self.lr = initial_lr
        self.DQN = DQN(self.input_shape, self.n_actions, gamma, initial_lr, initial_tau,
                       batch_size=batch_size, buffer_capacity=buffer_capacity,
                       n_envs=self.n_envs, n_step=n_step, double=double,
                       network=network, reward_scale=reward_scale, device=device)
        # One stack of frames per environment
        self.frame_stacks = [deque(maxlen=self.n_frames) for _ in range(self.n_envs)]
        print(f"DQN agent on device: {self.DQN.device}. Input shape: {self.input_shape}")
        

    def child_reset_episode(self, env_idx):
        rgb = self.pacman_envs[env_idx].rgb_image
        np_img = self.image_processing(rgb)
        tensor_img = torch.from_numpy(np_img)
        # (height, width, channels) -> (channels, height, width)
        if self.img_n_channels > 1:
            tensor_img = tensor_img.permute(2, 0, 1)
        self.episode_history.append(tensor_img)
        stack = []
        for i in range(self.n_frames):
            stack.append(tensor_img)
        self.image_frames = torch.stack(stack)
        current_state = self.frames_to_state()
        self.current_state.append(current_state)
        
        
    def epsilon_greedy(self, state, stockastic):
        def rand_argmax(tens):
            condition = (tens == tens.max()).flatten()
            max_inds = torch.argwhere(condition)
            return int(np.random.choice(max_inds.flatten()))

        p = np.random.random()
        if stockastic == False or p < (1 - self.epsilon):
            qvalues = self.DQN.qvalues(state).cpu()
            return rand_argmax(qvalues)
        return random.randint(0, self.n_actions - 1)
    
    
    def select_action(self, state, stockastic=False, env_idx=0):
        
        # Get the index of the action to perform according to current policy
        action_idx = self.epsilon_greedy(state, stockastic)
        action = self.actions[action_idx]
        return action_idx, action

        
    def take_action(self, env_idx):
        '''
        Play one move in one environment with the epsilon-greedy policy
        (no learning). Training uses run_episodes().
        '''
        state = self.current_state[env_idx]
        action_idx, action = self.select_action(state, stockastic=True, env_idx=env_idx)
        reward = self.pacman_envs[env_idx].step(action)
        self.episode_rewards.append(reward)
        np_img = self.image_processing(self.pacman_envs[env_idx].rgb_image)
        self.current_state[env_idx] = self.image_to_state(np_img)
        return reward


    def stacked_observation(self, env_idx, reset=False):
        '''
        State of environment 'env_idx': its last n_frames images stacked along
        the channel axis (oldest first). On reset, the first image is repeated.
        '''
        obs = self.observation(env_idx)
        stack = self.frame_stacks[env_idx]
        if reset:
            stack.clear()
            stack.extend([obs] * self.n_frames)
        else:
            stack.append(obs)
        return np.concatenate(stack, axis=0)
            
    
    # Run steps for training
    def run_episodes(self, total_steps=3_000_000, episode_max_length=1000,
                     epsilon_start=1.0, epsilon_end=0.02, epsilon_fraction=0.25,
                     learning_starts=20_000, train_freq=1, gradient_steps=1,
                     min_lr=None, log_interval=200, save_interval=250_000,
                     save_dir="./Tmp"):
        '''
        Train the agent.
        parameters:
            total_steps: total number of environment steps (transitions), all environments together.
            episode_max_length: games longer than this are aborted (truncated, not terminal).
            epsilon_start, epsilon_end, epsilon_fraction: epsilon decays linearly from
                epsilon_start to epsilon_end over the first epsilon_fraction * total_steps steps.
            learning_starts: number of transitions collected before learning starts.
            train_freq, gradient_steps: 'gradient_steps' updates every 'train_freq'
                vector steps (one vector step = one move in every environment).
            min_lr: if set, the learning rate decays linearly from initial_lr to min_lr.
            log_interval: print statistics every 'log_interval' finished games.
            save_interval: save the network every 'save_interval' steps.
        '''
        n_envs = self.n_envs
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.reset()
        states = []
        for env_idx in range(n_envs):
            self.reset_env(env_idx)
            states.append(self.stacked_observation(env_idx, reset=True))
        states = np.stack(states)

        ep_rewards = np.zeros(n_envs)
        ep_lengths = np.zeros(n_envs, dtype=np.int64)
        history = []
        loss_sum, q_sum, n_updates = 0., 0., 0
        lr = self.lr
        step, vec_step, next_save, logged = 0, 0, save_interval, 0
        start = time.time()
        last_log_time, last_log_step = start, 0

        while step < total_steps:
            # Linear epsilon schedule
            frac = min(1., step / max(1., epsilon_fraction * total_steps))
            self.epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

            # Batched epsilon-greedy action selection for all environments
            greedy = self.DQN.qvalues(states).argmax(dim=1).cpu().numpy()
            explore = np.random.random(n_envs) < self.epsilon
            actions = np.where(explore, np.random.randint(0, self.n_actions, n_envs), greedy)

            next_states = np.empty_like(states)
            for env_idx, env in enumerate(self.pacman_envs):
                reward = env.step(self.actions[actions[env_idx]])
                ep_rewards[env_idx] += reward
                ep_lengths[env_idx] += 1
                obs = self.stacked_observation(env_idx)
                done = env.end
                truncated = (not done) and ep_lengths[env_idx] >= episode_max_length
                self.DQN.memory.push(env_idx, states[env_idx], actions[env_idx], reward, obs, done, truncated)

                if done or truncated:
                    history.append({"step": step, "reward": ep_rewards[env_idx],
                                    "length": ep_lengths[env_idx],
                                    "won": bool(done and not env.dead), "lost": bool(env.dead),
                                    "aborted": bool(truncated), "ghosts_left": len(env.ghosts),
                                    "power_cookies_eaten": env.n_eaten_power_cookies})
                    self.reset_env(env_idx)
                    obs = self.stacked_observation(env_idx, reset=True)
                    ep_rewards[env_idx] = 0.
                    ep_lengths[env_idx] = 0
                next_states[env_idx] = obs

            states = next_states
            step += n_envs
            vec_step += 1

            # Learning
            if step >= learning_starts and vec_step % train_freq == 0:
                for _ in range(gradient_steps):
                    loss, q = self.DQN.learn()
                    loss_sum += loss
                    q_sum += q
                    n_updates += 1
                if min_lr is not None:
                    progress = (step - learning_starts) / max(1, total_steps - learning_starts)
                    lr = self.lr + min(1., progress) * (min_lr - self.lr)
                    self.DQN.set_lr(lr)

            # Logging
            if len(history) - logged >= log_interval:
                logged = len(history)
                recent = history[-log_interval:]
                now = time.time()
                sps = (step - last_log_step) / (now - last_log_time)
                last_log_time, last_log_step = now, step
                avg = lambda k: np.mean([h[k] for h in recent])
                loss_str = (f"loss {float(loss_sum)/n_updates:.4f}, Q {float(q_sum)/n_updates:.2f}, "
                            if n_updates else "")
                print(f"{timedelta(seconds=int(now - start))} | step {step:>9} | games {len(history):>7} | "
                      f"eps {self.epsilon:.3f} | reward {avg('reward'):7.2f} | "
                      f"won {100*avg('won'):5.1f}% lost {100*avg('lost'):5.1f}% aborted {100*avg('aborted'):4.1f}% | "
                      f"ghosts left {avg('ghosts_left'):.2f} | len {avg('length'):5.1f} | "
                      f"{loss_str}lr {lr:.2e} | {sps:.0f} steps/s", flush=True)
                loss_sum, q_sum, n_updates = 0., 0., 0

            # Checkpoints
            if step >= next_save:
                next_save += save_interval
                self.save(str(save_dir / f"pacman_dqn_{step}.pth"))
                pd.DataFrame(history).to_csv(save_dir / "pacman_dqn_training_log.csv", index=False)

        self.save(str(save_dir / "pacman_dqn_final.pth"))
        pd.DataFrame(history).to_csv(save_dir / "pacman_dqn_training_log.csv", index=False)
        print(f"Training done: {step} steps, {len(history)} games, {timedelta(seconds=int(time.time() - start))}")
        return history

    def save(self, filename):
        self.DQN.save(filename)
        
    def load(self, filename):
        self.DQN.load(filename)
        # If the checkpoint had another architecture the network was rebuilt
        self.input_shape = self.DQN.input_shape




class A2C_PacmanAgent(PacmanAgent):
    def __init__(self, pacman_envs: List[PacmanEnvironment], gamma=0.99, initial_lr=0.0005, grayscale=False, n_frames=3, n_steps = 5):
        super().__init__(pacman_envs, n_frames, grayscale)
        self.lr = initial_lr
        self.n_steps = n_steps
        
        self.A2C = A2C(self.input_shape, self.pacman_envs[0].n_actions, self.n_steps, self.n_envs, gamma, initial_lr)

    def epsilon_greedy(self, state, eps):
        pass
        
    def child_reset_episode(self, env_idx):
        rgb = self.pacman_envs[env_idx].rgb_image
        np_img = self.image_processing(rgb)
        tensor_img = torch.from_numpy(np_img)
        # (height, width, channels) -> (channels, height, width)
        if self.img_n_channels > 1:
            tensor_img = tensor_img.permute(2, 0, 1)
        self.episode_history.append(tensor_img)
        stack = []
        for i in range(self.n_frames):
            stack.append(tensor_img)
        self.image_frames = torch.stack(stack)
        current_state = self.frames_to_state()
        self.current_state.append(current_state)
        self.A2C.reset_memory(env_idx, current_state)
        #print(f"Reset episode env {env_idx}. State: {current_state}")
            
    
    def select_action(self, state, stockastic=False, env_idx=0):
        # No gradient needed here: the losses are recomputed in A2C.evaluate_actions()
        with torch.no_grad():
            logit, value = self.A2C.a2c_net(state.to(self.A2C.device))
            probs = F.softmax(logit, dim=-1)
        #print(f"logit: {logit}. probs: {probs}")
        
        if stockastic:
            action_idx = probs.multinomial(num_samples=1)[0,0].item()
        else:
            action_idx = torch.argmax(probs).item()
            
        action = self.actions[action_idx]
        
        return action_idx, action

        
    def take_action(self, env_idx):
        # get the current state
        state = self.current_state[env_idx]
        
        #print(f"env {env_idx}. state: {state}")
        #print(f"mem state: {self.A2C.memory[env_idx].get_last_state()}")
        
        #assert torch.equal(self.A2C.memory[env_idx].get_last_state(), state)
        
        # Take action according to current policy
        action_idx, action = self.select_action(state, stockastic=True,
                                                env_idx=env_idx)
        #print(f"Action index: {action_idx} -> action: {action}")
        
        reward = self.pacman_envs[env_idx].step(action)
        self.episode_rewards.append(reward)

        # get the reached state
        rgb = self.pacman_envs[env_idx].rgb_image
        np_img = self.image_processing(rgb)
        # Put in the format for Pytorch Conv2d
        new_state = self.image_to_state(np_img)
        
        # Update current state
        self.current_state[env_idx] = new_state

        # Add a new entry in the rollout memory
        self.A2C.memory[env_idx].insert(new_state,
                                        torch.Tensor([action_idx]).type(torch.int64),
                                        reward)

        
    def run_episodes(self, episode_max_length=500, total_episodes=1000, min_lr=0.00002):

        #print(f"Number of env: {self.n_envs}")
        
        def learning_rate_generator(epochs, min_lr):
            updated_lr = self.lr
            epoch = 0
            while epoch < epochs:
                if updated_lr < min_lr:
                    updated_lr = min_lr
                yield updated_lr
                epoch += 1
                updated_lr = self.lr * ( 1 - (epoch / epochs))**2
        
        lr = learning_rate_generator(total_episodes, min_lr)

        self.reset()
        episodes = 0
        self.reward_per_episode = []
        episodes_durations = []
        average_duration_str = ""
        average_reward_str = ""
        lost_count = 0
        ghost_count = 0
        finished_games = 1
        alpha = self.lr
        #fig, ax = plt.subplots(1,1)
        
        training_start = time.time()
        
        while episodes != total_episodes:
            start = time.time()
                
            # Reset for an episode
            self.reset_episode()
            
            # For the first 10000 episodes the behavior of the edible ghosts
            # is set to RANDOM
            # if episodes < 10000:
            #     for env in self.pacman_envs:
            #         for ghost in env.ghosts:
            #             ghost.edible_ghost_behavior = EdibleGhostBehaviors.RANDOM
                    

            # Run an episode in each environment
            current_episode_counter = episode_max_length
            while True:
                value_losses = []
                policy_losses = []
                
                #assert torch.equal(self.current_state, self.A2C.memory.states[0])
                for env_idx in range(self.n_envs):
                    if self.pacman_envs[env_idx].end == True:
                        continue
                    # Take up to n_steps in each environment
                    for step in range(self.n_steps):
                        self.take_action(env_idx)
                        #print(f"Memory after action {step}")
                        #self.A2C.memory.show()
                        env = self.pacman_envs[env_idx]
                        if env.end == True:
                            lost_count += int(env.dead == True)
                            ghost_count += len(env.ghosts)
                            finished_games += 1
                            break

                    #assert torch.equal(self.current_state, self.A2C.memory.get_last_state())
                    value_loss, policy_loss = self.A2C.compute_losses(env_idx, step+1,
                                                                      self.pacman_envs[env_idx].end)
                    value_losses.append(value_loss)
                    policy_losses.append(policy_loss)
                
                self.A2C.update(value_losses, policy_losses)

                ended_envs = 0
                for env_idx in range(self.n_envs):
                    if self.pacman_envs[env_idx].end == True:
                        ended_envs += 1
                        continue
                    self.A2C.roll_memory(env_idx)

                if ended_envs == self.n_envs:
                    break

                current_episode_counter -= 1
                if current_episode_counter  == 0:
                    # Abort too long games
                    break

            self.reward_per_episode.append(np.sum(self.episode_rewards))

            if alpha != min_lr:
                alpha = next(lr)
                for param_group in self.A2C.optimizer.param_groups:
                    param_group['lr'] = alpha
            
            episodes += 1
            
            end = time.time()
            episodes_durations.append(timedelta(seconds=end-start))

            if episodes % 100 == 0:
                average_duration_str = f"Avg duration: {np.mean(episodes_durations)}"
                average_reward_str = f", Avg reward: {np.mean(self.reward_per_episode)/self.n_envs:.2f}"
                average_ghosts = float(ghost_count)/finished_games
                print(f"{timedelta(seconds=time.time() - training_start)}. Episodes {episodes}. " + 
                      average_duration_str + average_reward_str + f", lost {float(lost_count*100)/finished_games:.1f}%. " +
                      f"Avg ghost: {average_ghosts:.3f}, (lr: {alpha:.6f})")
                episodes_durations = []
                lost_count = 0
                ghost_count = 0
                finished_games = 0
                
                df = pd.DataFrame({"rewards": self.reward_per_episode, "ghosts": average_ghosts})
                df.to_csv('./Tmp/pacman_a2c_reward_per_episode.csv', mode='a', index=False, header=False)
                self.reward_per_episode = []
        
            if episodes % 5000 == 0 or episodes == total_episodes:
                print(f"Save NN params (episodes: {episodes})")
                filename = f"./Tmp/pacman_color_a2c_{episodes}.pth"
                self.save(filename)
        

    def save(self, filename):
        self.A2C.save(filename)
        
    def load(self, filename):
        self.A2C.load(filename)