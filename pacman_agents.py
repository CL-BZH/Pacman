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
            
            # Select a start position
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
    
            self.episode_history = []
            self.episode_rewards = []
            
            # Child class specific reset
            self.child_reset_episode(env_idx)
            
        
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
    def __init__(self, pacman_envs: List[PacmanEnvironment], gamma=0.99, initial_lr=0.0005, initial_tau=1., grayscale=False, n_frames=1):
        super().__init__(pacman_envs, n_frames, grayscale)
        self.epsilon = 1
        self.lr = initial_lr
        self.DQN = DQN(self.input_shape, self.n_actions, gamma, initial_lr, initial_tau)
        

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
        #print(f"{env_idx}. Reset. Current state: {current_state}.\n Shape: {current_state.shape}")
        self.current_state.append(current_state)
        
        
    def epsilon_greedy(self, state, stockastic):
        def rand_argmax(tens):
            condition = (tens == tens.max()).flatten()
            max_inds = torch.argwhere(condition)
            return np.random.choice(max_inds.flatten())

        p = np.random.random()
        if stockastic == False or p < (1 - self.epsilon):
            with torch.no_grad():
                qvalues = self.DQN.policy_net(state) # Call CNN.forward(state)
                #action = torch.argmax(qvalues).detach().cpu().numpy()
                action = rand_argmax(qvalues)
                #print(f"qvalues: {qvalues}. action: {action}\n")
                return action
        return random.randint(0, self.n_actions - 1)
    
    
    def select_action(self, state, stockastic=False, env_idx=0):
        
        # Get the index of the action to perform according to current policy
        action_idx = self.epsilon_greedy(state, stockastic)
        action = self.actions[action_idx]
        return action_idx, action

        
    def take_action(self, env_idx):
        # get the current state
        state = self.current_state[env_idx]
        #print(f"Current state: {state}")
                
        # Take action according to current policy
        action_idx, action = self.select_action(state, stockastic=True, env_idx=env_idx)
        #print(f"actions: {self.actions}. action idx: {action_idx}")
         
        # Pass the action to the environment and get a reward
        reward = self.pacman_envs[env_idx].step(action)
        self.episode_rewards.append(reward)
        
        # get the reached state
        rgb = self.pacman_envs[env_idx].rgb_image
        np_img = self.image_processing(rgb)
        # Put in the format for Pytorch Conv2d
        new_state = self.image_to_state(np_img)
        
        # Update current state
        self.current_state[env_idx] = new_state
        
        # Add a new entry in the Replay memory
        self.DQN.memory.push((state,
                              torch.Tensor([action_idx]).type(torch.int64),
                              torch.Tensor([reward]),
                              new_state,
                              self.pacman_envs[env_idx].end))
            
    
    # Run episodes for training
    def run_episodes(self, episode_max_length=500, total_episodes=10000, epsilon=1, decay=0.9999, min_lr=0.0002, r=0.9):

        def epsilon_generator(r):
            eps_start = self.epsilon
            eps_end = 0.01
            eps = eps_start
            e = eps
            s = (eps_end - 0.5)/total_episodes
            episode = 0
            while episode < total_episodes:
                yield eps
                episode += 1
                y = s * episode + 0.5
                e *= decay
                #e = max(e, eps_end)
                eps = r * e + (1-r) * y
            
        def learning_rate_generator():
            updated_lr = self.lr
            episode = 0
            while episode < total_episodes:
                if updated_lr < min_lr:
                    updated_lr = min_lr
                yield updated_lr
                episode += 1
                updated_lr = self.lr * ( 1 - (episode / total_episodes))**4
        
        self.epsilon = epsilon
        eps_gen = epsilon_generator(r)
        
        if min_lr != self.lr:
            lr = learning_rate_generator()
        
        self.reset()
        episodes = 0
        self.reward_per_episode = []
        episodes_durations = []
        average_duration_str = ""
        average_reward_str = ""
        lost_percentage_str = ""
        back = -99
        lost_count = 0
        ghost_count = 0
        finished_games = 1
        aborted_game = False
        rolling = 1000
        alpha = self.lr

        
        while episodes != total_episodes:
            start = time.time()
            
            if episodes % 100 == 0:
                print(f"Episodes {episodes}. Epsilon = {self.epsilon:.3f}" + 
                      average_duration_str + average_reward_str +
                      lost_percentage_str +
                      f". Avg ghost: {float(ghost_count)/finished_games:.3f}." +
                      f", (lr: {alpha:.6f})")
                episodes_durations = []
                lost_count = 0
                finished_games = 0
                ghost_count = 0
            
            if episodes % (5*rolling) == 0:
                print(f"Save NN params (episodes: {episodes})")
                filename = f"./Tmp/pacman_color_dqn_{episodes}.pth"
                self.save(filename)
                
                column_name = f"Reward per episode on {self.n_envs} environments"
                df = pd.DataFrame(self.reward_per_episode, columns = [column_name,])
                
                #df.to_csv('./Tmp/pacman_dqn_cum_avg_reward_per_episode.csv', index=False)
                df.to_pickle('./Tmp/pacman_dqn_cum_avg_reward_per_episode.pkl')
                
            # Reset for an episode
            self.reset_episode()
            
            # Reduce the number of environment
            if episodes == total_episodes // 2:
                self.n_envs = max(1, self.n_envs // 2)
                print(f"Episodes {episodes}. -> Number of environment divided by 2 (self.n_envs = {self.n_envs})")
            
            # Run an episode in each environment
            for env_idx in range(self.n_envs):
                #assert self.pacman_envs[env_idx].end == False
                
                self.episode_rewards = []
                current_episode_counter = episode_max_length
                env = self.pacman_envs[env_idx]
                while env.end == False:
                    current_episode_counter -=1
                    self.take_action(env_idx)
                    if (current_episode_counter == 0 and env.end == False):
                        #print(f"Abort env {env_idx}. Number of ghost: {len(self.pacman_envs[env_idx].ghosts)} !")
                        # Abort the game for this environment
                        #self.episode_rewards[-1] = -5
                        aborted_game = True
                        break
                        
                #print(f"Env {env_idx}. Total reward: {np.sum(self.episode_rewards)}")
                self.reward_per_episode.append(np.sum(self.episode_rewards))
    
                lost_count += int(self.pacman_envs[env_idx].dead == True)
                
                if aborted_game == False:
                    finished_games += 1
                    ghost_count += len(env.ghosts)
                    
                aborted_game = False
            
            
            # Update the Neural Network
            self.DQN.update()
            
            # If use learning rate decay:
            if min_lr != self.lr:
                alpha = next(lr)
                for param_group in self.DQN.optimizer.param_groups:
                    param_group['lr'] = alpha
            
            episodes += 1
            self.epsilon = next(eps_gen)
            
            lost_percentage = lost_count*100/(finished_games+0.0000001)
            
            end = time.time()
            episodes_durations.append(timedelta(seconds=end-start))
            average_duration_str = f", Average duration: {np.mean(episodes_durations)}"
            back = max(-len(self.reward_per_episode)+1, -99)
            average_reward_str = f", Average reward: {np.mean(self.reward_per_episode[back:-1]):.2f}"
            lost_percentage_str = f". lost: {lost_percentage:.1f}%"    

        print(f"Episodes {episodes}. Epsilon = {self.epsilon:.3f}, "
              f", Average duration: {np.mean(episodes_durations)}"
              f", Average reward: {np.mean(self.reward_per_episode[back:-1]):.2f}")

    def save(self, filename):
        self.DQN.save(filename)
        
    def load(self, filename):
        self.DQN.load(filename)




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
        logit, value = self.A2C.a2c_net(state)
        probs = F.softmax(logit, dim=-1)
        #print(f"logit: {logit}. probs: {probs}")
        
        if stockastic:
            action_idx = probs.multinomial(num_samples=1).data[0,0]
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