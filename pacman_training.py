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
    
    initial_lr = 0.00001
    min_lr = 0.00001
    episodes = 60000
    
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
    
    
def dqn_training():
    n_envs = 32
    
    initial_lr = 0.0005
    min_lr = 0.0005
    initial_tau = 1.
    initial_epsilon = 0.99
    epsilon_decay = 0.9999
    r = 0.93
    
    # Second training (no forced learning rate decay)
    # initial_lr = 0.00008
    # min_lr = 0.00008
    # initial_tau = 0.1
    # initial_epsilon = 0.05
    # epsilon_decay=0.99995
    # r = 0.5
    
    episodes = 300000
    
    dqn_pacman_envs = []
    for _ in range(n_envs):
        env = PacmanEnvironment(pacman_conf)
        #env.edible_ghosts_behavior = [EdibleGhostBehaviors.RANDOM] * pacman_conf.n_ghosts
        dqn_pacman_envs.append(env)

    dqn_pacman_agent = DQN_PacmanAgent(dqn_pacman_envs, gamma=0.99, initial_lr=initial_lr, initial_tau=initial_tau)#, grayscale=False, n_frames=1)

    start = time.time()

    dqn_pacman_agent.run_episodes(episode_max_length=1000, total_episodes=episodes, epsilon=initial_epsilon, decay=epsilon_decay, min_lr=min_lr, r=r)

    end = time.time()
    print(f"Elapsed time: {timedelta(seconds=end-start)}")
    dqn_pacman_agent.save("pacman_color_dqn.pth")
    
    
if __name__ == "__main__":
    
    Path("./Tmp").mkdir(parents=True, exist_ok=True)
    
    a2c_training("./Trainings/A2C/Config_2/states_dict_55000.pth")
    #a2c_training()
    