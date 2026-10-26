#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:38:31 2024

@author: chris
"""

from pacman_env import PacmanEnvironment
from algorithms import DQN, A2C
import random
import numpy as np
from scipy.spatial.distance import cdist
import time
from datetime import timedelta
from abc import ABC, abstractmethod
import torch 
from typing import List
import pandas as pd
import pathlib
from collections import deque


class PacmanAgent(ABC):
    '''
    Base class of the agents.

    The state given to the networks is made of the last 'n_frames' images of
    an environment stacked along the channel axis (one stack per environment).
    The image is either the environment's RGB(+power) image, optionally in
    grayscale, or its feature planes (features=True, see
    PacmanEnvironment.feature_image).
    '''
    def __init__(self, pacman_envs: List[PacmanEnvironment], n_frames=1, grayscale=False, features=False):
        self.pacman_envs = pacman_envs
        self.n_envs = len(pacman_envs)
        self.weights = np.ones(len(self.pacman_envs[0].valid_pos))
        self.actions = self.pacman_envs[0].get_all_actions()
        self.n_actions = len(self.actions)
        self.features = features
        self.grayscale = grayscale and not features
        self.n_frames = n_frames

        env = self.pacman_envs[0]
        rows, columns, channels = env.feature_image_shape if features else env.rgb_image_shape
        self.img_n_channels = 1 if self.grayscale else channels
        self.input_shape = (self.img_n_channels * self.n_frames, rows, columns)

        self.frame_stacks = [deque(maxlen=self.n_frames) for _ in range(self.n_envs)]
        self.current_state = [None] * self.n_envs
        self.episode_rewards = []

    def reset(self):
        for i in range(self.n_envs):
            self.pacman_envs[i].reset()
        self.current_state = [None] * self.n_envs
        self.episode_rewards = []

    def reset_episode(self):
        '''
        Start a new game in every environment.
        '''
        for env_idx in range(self.n_envs):
            self.reset_env(env_idx)
            self.current_state[env_idx] = self.observe(env_idx, reset=True)
        self.episode_rewards = []

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
        env = self.pacman_envs[env_idx]
        img = env.feature_image if self.features else env.rgb_image
        if self.grayscale:
            scale = np.array([0.2989, 0.5870, 0.1140], dtype='float32')
            img = np.dot(img[..., :3], scale)[..., np.newaxis]
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32)

    def stacked_observation(self, env_idx, reset=False):
        '''
        State of environment 'env_idx': its last n_frames images stacked along
        the channel axis (oldest first). On reset, the first image is repeated.
        Must be called once after each move (it pushes the new image).
        '''
        obs = self.observation(env_idx)
        stack = self.frame_stacks[env_idx]
        if reset:
            stack.clear()
            stack.extend([obs] * self.n_frames)
        else:
            stack.append(obs)
        return np.concatenate(stack, axis=0)

    def observe(self, env_idx=0, reset=False):
        '''
        Same as stacked_observation() but as a (1, channels, height, width)
        tensor; also updates current_state. Used to play games.
        '''
        state = torch.from_numpy(self.stacked_observation(env_idx, reset)).unsqueeze(0)
        self.current_state[env_idx] = state
        return state

    @abstractmethod
    def select_action(self, state, stockastic=False, env_idx=0):
        '''
        Returns (action index, action) for a state of shape (1, C, H, W).
        '''
        pass

    @abstractmethod
    def run_episodes(self, **kwargs):
        '''
        Training.
        '''
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
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
        print(f"DQN agent on device: {self.DQN.device}. Input shape: {self.input_shape}")
        
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
        return action_idx, self.actions[action_idx]

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



class A2C_PacmanAgent(PacmanAgent):
    '''
    Actor-critic agent (A2C, or PPO with ppo_epochs > 1).

    - Input: the environment's feature planes (walls, cookies, power cookies,
      Pacman, one plane per ghost, power level), 2 stacked frames by default
      so that the direction in which the ghosts move is visible.
    - Network: full-resolution actor-critic CNN, made exactly equivariant to the
      left-right mirror symmetry of the board when the board is symmetric
      (symmetric=True).
    - Training (run_episodes): all environments step in lockstep for n_steps,
      then advantages are computed with GAE and the network is updated with
      PPO (or A2C when ppo_epochs=1 and n_minibatches=1).
    - Optional curriculum: the ghosts' aggressiveness starts at
      'curriculum_start' times its configured value and increases linearly to
      the full value over the first 'curriculum_fraction' of the training.
    '''
    def __init__(self, pacman_envs: List[PacmanEnvironment], gamma=0.99, initial_lr=2.5e-4, n_frames=2,
                 gae_lambda=0.95, entropy_coef=0.03, value_loss_coef=0.5, ppo_epochs=4, n_minibatches=4,
                 clip_eps=0.2, reward_scale=0.1, symmetric=True, device=None):
        super().__init__(pacman_envs, n_frames, grayscale=False, features=True)
        self.lr = initial_lr

        action_mirror = None
        board = self.pacman_envs[0].map.map
        if symmetric:
            if np.array_equal(board, board[:, ::-1]):
                mirror_name = {'E': 'W', 'W': 'E', 'N': 'N', 'S': 'S', 'I': 'I'}
                action_mirror = [self.actions.index(mirror_name[a]) for a in self.actions]
            else:
                print("The board is not left-right symmetric: symmetric network disabled.")

        self.A2C = A2C(self.input_shape, self.n_actions, gamma=gamma, initial_lr=initial_lr,
                       gae_lambda=gae_lambda, entropy_coef=entropy_coef, value_loss_coef=value_loss_coef,
                       ppo_epochs=ppo_epochs, n_minibatches=n_minibatches, clip_eps=clip_eps,
                       reward_scale=reward_scale, action_mirror=action_mirror, device=device)
        print(f"A2C agent on device: {self.A2C.device}. Input shape: {self.input_shape}. "
              f"Symmetric network: {action_mirror is not None}")

    def select_action(self, state, stockastic=False, env_idx=0):
        actions, _, _ = self.A2C.act(state, stochastic=stockastic)
        action_idx = int(actions[0])
        return action_idx, self.actions[action_idx]

    def run_episodes(self, total_steps=20_000_000, n_steps=128, episode_max_length=1000,
                     min_lr=None, curriculum_fraction=0., curriculum_start=0.3,
                     log_interval=500, save_interval=1_000_000, save_dir="./Tmp"):
        '''
        Train the agent.
        parameters:
            total_steps: total number of environment steps (transitions), all environments together.
            n_steps: rollout length per environment between two updates
                (an update uses n_steps * n_envs transitions).
            episode_max_length: games longer than this are aborted (truncated, not terminal).
            min_lr: if set, the learning rate decays linearly from initial_lr to min_lr.
            curriculum_fraction, curriculum_start: see the class description (0 = no curriculum).
            log_interval: print statistics every 'log_interval' finished games.
            save_interval: save the network every 'save_interval' steps.
        '''
        n_envs = self.n_envs
        T = n_steps
        gamma = self.A2C.gamma
        reward_scale = self.A2C.reward_scale
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        base_levels = [list(env.ghost_aggressiveness_levels) for env in self.pacman_envs]

        def curriculum_scale(step):
            if curriculum_fraction <= 0:
                return 1.
            return curriculum_start + (1. - curriculum_start) * min(1., step / (curriculum_fraction * total_steps))

        def new_game(env_idx, scale):
            # Each environment gets its own list (the config's list is shared)
            self.pacman_envs[env_idx].ghost_aggressiveness_levels = [a * scale for a in base_levels[env_idx]]
            self.reset_env(env_idx)
            return self.stacked_observation(env_idx, reset=True)

        scale = curriculum_scale(0)
        states = np.stack([new_game(i, scale) for i in range(n_envs)])

        buf_states = np.zeros((T, n_envs, *self.input_shape), dtype=np.float32)
        buf_actions = np.zeros((T, n_envs), dtype=np.int64)
        buf_log_probs = np.zeros((T, n_envs), dtype=np.float32)
        buf_values = np.zeros((T, n_envs), dtype=np.float32)
        buf_rewards = np.zeros((T, n_envs), dtype=np.float32)
        buf_dones = np.zeros((T, n_envs), dtype=np.float32)

        ep_rewards = np.zeros(n_envs)
        ep_lengths = np.zeros(n_envs, dtype=np.int64)
        history = []
        update_stats = []
        lr = self.lr
        step, next_save, logged = 0, save_interval, 0
        start = time.time()
        last_log_time, last_log_step = start, 0

        while step < total_steps:
            scale = curriculum_scale(step)

            # Collect a rollout of T steps in every environment
            for t in range(T):
                actions, log_probs, values = self.A2C.act(states)
                buf_states[t] = states
                buf_actions[t] = actions
                buf_log_probs[t] = log_probs
                buf_values[t] = values

                next_states = np.empty_like(states)
                for env_idx, env in enumerate(self.pacman_envs):
                    reward = env.step(self.actions[actions[env_idx]])
                    ep_rewards[env_idx] += reward
                    ep_lengths[env_idx] += 1
                    obs = self.stacked_observation(env_idx)
                    done = env.end
                    truncated = (not done) and ep_lengths[env_idx] >= episode_max_length

                    r = reward * reward_scale
                    if truncated:
                        # Aborted game: not a real end, bootstrap from the value of the reached state
                        r += gamma * self.A2C.values(obs[np.newaxis])[0]
                    buf_rewards[t, env_idx] = r
                    buf_dones[t, env_idx] = float(done or truncated)

                    if done or truncated:
                        history.append({"step": step, "reward": ep_rewards[env_idx],
                                        "length": ep_lengths[env_idx],
                                        "won": bool(done and not env.dead), "lost": bool(env.dead),
                                        "aborted": bool(truncated), "ghosts_left": len(env.ghosts),
                                        "power_cookies_eaten": env.n_eaten_power_cookies,
                                        "aggressiveness_scale": scale})
                        obs = new_game(env_idx, scale)
                        ep_rewards[env_idx] = 0.
                        ep_lengths[env_idx] = 0
                    next_states[env_idx] = obs
                states = next_states
                step += n_envs

            # Advantages and update
            advantages, returns = self.A2C.compute_gae(buf_rewards, buf_values, buf_dones,
                                                       self.A2C.values(states))
            update_stats.append(self.A2C.update(buf_states.reshape(T * n_envs, *self.input_shape),
                                                buf_actions.reshape(-1), buf_log_probs.reshape(-1),
                                                advantages.reshape(-1), returns.reshape(-1)))
            if min_lr is not None:
                lr = self.lr + min(1., step / total_steps) * (min_lr - self.lr)
                self.A2C.set_lr(lr)

            # Logging
            if len(history) - logged >= log_interval:
                logged = len(history)
                recent = history[-log_interval:]
                now = time.time()
                sps = (step - last_log_step) / (now - last_log_time)
                last_log_time, last_log_step = now, step
                avg = lambda k: np.mean([h[k] for h in recent])
                u = {k: np.mean([s[k] for s in update_stats]) for k in update_stats[0]}
                update_stats = []
                curriculum_str = f"aggr x{scale:.2f} | " if curriculum_fraction > 0 else ""
                print(f"{timedelta(seconds=int(now - start))} | step {step:>9} | games {len(history):>7} | "
                      f"{curriculum_str}reward {avg('reward'):7.2f} | "
                      f"won {100*avg('won'):5.1f}% lost {100*avg('lost'):5.1f}% aborted {100*avg('aborted'):4.1f}% | "
                      f"ghosts left {avg('ghosts_left'):.2f} | len {avg('length'):5.1f} | "
                      f"entropy {u['entropy']:.3f}, v_loss {u['value_loss']:.3f}, kl {u['approx_kl']:.4f}, "
                      f"clip {u['clip_frac']:.2f} | lr {lr:.2e} | {sps:.0f} steps/s", flush=True)

            # Checkpoints
            if step >= next_save:
                next_save += save_interval
                self.save(str(save_dir / f"pacman_a2c_{step}.pth"))
                pd.DataFrame(history).to_csv(save_dir / "pacman_a2c_training_log.csv", index=False)

        # Restore the configured aggressiveness
        for env, levels in zip(self.pacman_envs, base_levels):
            env.ghost_aggressiveness_levels = levels
        self.save(str(save_dir / "pacman_a2c_final.pth"))
        pd.DataFrame(history).to_csv(save_dir / "pacman_a2c_training_log.csv", index=False)
        print(f"Training done: {step} steps, {len(history)} games, {timedelta(seconds=int(time.time() - start))}")
        return history

    def save(self, filename):
        self.A2C.save(filename)
        
    def load(self, filename):
        self.A2C.load(filename)
