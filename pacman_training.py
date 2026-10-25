#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:32:36 2024

@author: chris
"""

from pacman_maps import pacman_map1
from pacman_maps import pacman_map2
from pacman_maps import pacman_map3
from pacman_conf import PacmanConf
from pacman_env import PacmanEnvironment
from ghost_agents import EdibleGhostBehaviors
from pacman_agents import DQN_PacmanAgent
from pacman_agents import A2C_PacmanAgent
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path


pacman_conf = PacmanConf(playground = pacman_map3,
                            ghost_aggressiveness_levels = [0.74, 0.64], #, 0.52, 0.42],
                            edible_ghosts_behavior = [EdibleGhostBehaviors.FLEE_FAST, EdibleGhostBehaviors.FLEE_FAST],
                            n_ghosts = 2,
                            n_power_cookies = 2,
                            max_power_credit = 12,
                            eat_cookie_reward = 0.8,
                            eat_power_cookie_reward = 1.3,
                            eat_ghost_reward = 20,
                            living_cost = -0.15,
                            lose_reward = -35,
                            win_reward = 30,
                            penalty_per_ghost = -5)

def a2c_training(state_dict=None):
    n_envs = 21
    n_steps = 19
    
    initial_lr = 0.0004
    min_lr = 0.00001
    episodes = 300000
    
    a2c_pacman_envs = []
    for _ in range(n_envs):
        a2c_pacman_envs.append(PacmanEnvironment(pacman_conf))#, start_ghosts_pos=[(5,6), (5,8)]))
    
    a2c_pacman_agent = A2C_PacmanAgent(a2c_pacman_envs, initial_lr=initial_lr, grayscale=False, n_frames=1, n_steps=n_steps)
    if state_dict is not None:
        a2c_pacman_agent.load(state_dict)
    
    start = time.time()
    a2c_pacman_agent.run_episodes(episode_max_length=600, total_episodes=episodes, min_lr=min_lr)
    end = time.time()
    print(f"Elapsed time: {timedelta(seconds=end-start)}")
    path = './Trainings/A2C/'
    new_dir_name = f"{end}"
    new_dir = pathlib.Path(path, new_dir_name)
    new_dir.mkdir()
    pth = path + new_dir_name + "/" + "state_dict.pth"
    a2c_pacman_agent.save(pth)
    conf_file = new_dir / 'config.txt'
    t1 = f"pacman_conf = {pacman_conf}"
    t2 = f"n_envs = {n_envs}"
    t3 = f"n_steps = {n_steps}"
    t4 = f"initial_lr = {initial_lr}"
    t5 = f"min_lr = {min_lr}"
    t6 = f"episodes = {episodes}"
    conf_file.write_text("\n".join([t1,t2,t3,t4,t5,t6]))
    
    
def dqn_training(state_dict=None, total_steps=5_000_000):
    n_envs = 16           # environments stepped in lockstep (batched forward pass)
    n_frames = 2          # 2 stacked frames: lets the network see in which direction the ghosts move
    
    initial_lr = 2.5e-4
    min_lr = 5e-5         # linear decay of the learning rate down to min_lr
    tau = 0.005           # Polyak averaging for the target network
    n_step = 3            # n-step returns
    batch_size = 128
    buffer_capacity = 300_000
    
    dqn_pacman_envs = [PacmanEnvironment(pacman_conf) for _ in range(n_envs)]

    dqn_pacman_agent = DQN_PacmanAgent(dqn_pacman_envs, gamma=0.99, initial_lr=initial_lr, initial_tau=tau,
                                       grayscale=False, n_frames=n_frames, batch_size=batch_size,
                                       buffer_capacity=buffer_capacity, n_step=n_step,
                                       double=True, network="dueling", reward_scale=0.1)
    if state_dict is not None:
        dqn_pacman_agent.load(state_dict)

    start = time.time()
    dqn_pacman_agent.run_episodes(total_steps=total_steps, episode_max_length=1000,
                                  epsilon_start=1.0, epsilon_end=0.02, epsilon_fraction=0.25,
                                  learning_starts=20_000, train_freq=1, gradient_steps=1,
                                  min_lr=min_lr, log_interval=200, save_interval=250_000,
                                  save_dir="./Tmp")
    end = time.time()
    print(f"Elapsed time: {timedelta(seconds=end-start)}")

    new_dir = pathlib.Path('./Trainings/DQN/', f"{end}")
    new_dir.mkdir(parents=True)
    dqn_pacman_agent.save(str(new_dir / "state_dict.pth"))
    (new_dir / 'config.txt').write_text("\n".join([
        f"pacman_conf = {pacman_conf}", f"n_envs = {n_envs}", f"n_frames = {n_frames}",
        f"initial_lr = {initial_lr}", f"min_lr = {min_lr}", f"tau = {tau}", f"n_step = {n_step}",
        f"batch_size = {batch_size}", f"buffer_capacity = {buffer_capacity}",
        f"total_steps = {total_steps}"]))
    
    
if __name__ == "__main__":
    
    Path("./Tmp").mkdir(parents=True, exist_ok=True)
    
    #a2c_training("./Trainings/A2C/Config_2/states_dict_55000.pth")
    a2c_training()
    #dqn_training()
    