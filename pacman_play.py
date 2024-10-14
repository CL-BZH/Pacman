#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:24:25 2024

@author: chris
"""

from pacman_maps import pacman_map2
from pacman_maps import pacman_map3
from pacman_conf import PacmanConf
from pacman_env import PacmanEnvironment
from ghost_agents import EdibleGhostBehaviors
from pacman_agents import DQN_PacmanAgent
from pacman_agents import A2C_PacmanAgent
from matplotlib import pyplot as plt
import numpy as np
import time
from collections import defaultdict


def play_game(env, agent, stockastic=False, display=False):

    def set_figure(env: PacmanEnvironment):
        rows = env.state[0].shape[0]
        columns = env.state[0].shape[1]
        m = max(rows, columns)
        width = 5 * float(columns)/m
        height = 5 * float(rows)/m
        plt.ion()
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(width, height)
        ax.set_axis_off()
        return width, height, fig, ax

    
    agent.reset_episode()
        
    if display == True:
        width, height, fig, ax = set_figure(env)
        env.colormap(width, height, fig, ax)
        plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 
        plt.pause(15)
        ax.cla()
        
    rewards = []
    aborted = False
    
    while env.end == False:
        rgb = agent.pacman_envs[0].rgb_image
        np_img = agent.image_processing(rgb)
        state = agent.image_to_state(np_img)
        if env.ghosts != []:
            _, action = agent.select_action(state, stockastic=False)
        else:
            _, action = agent.select_action(state, stockastic=stockastic)

        reward = env.step(action)
        agent.episode_rewards.append(reward)
        rewards.append(reward)
        
        if display == True:
            env.colormap(width, height, fig, ax)
            if env.end == False:
                ax.cla()
            else:
                plt.ioff()
                plt.show()
                
        if len(rewards) == 600:
            print(f"\nNon finished. Number of ghost: {len(env.ghosts)}.") #"\nRewards:\n{rewards}")
            #display = True
            if display == True:
                width, height, fig, ax = set_figure(env)
                env.colormap(width, height, fig, ax)
                time.sleep(0.7)
                plt.close();
        if len(rewards) > 800:
            print(f"\nAbort.Number of ghost: {len(env.ghosts)}.") #"\nRewards:\n{rewards}")
            aborted = True
            break
        
    # Return True if Pacman lost the game
    #print(f"Last reward: {rewards[-1]}")
    return env.dead == True, aborted, len(env.ghosts), env.n_eaten_power_cookies


n_games = 1#000
lost = []
abort_count = defaultdict(int)
ghost_count = defaultdict(int)
eaten_power_cookies_count = defaultdict(int)
display = True
if n_games > 1:
    display = False

pacman_conf = PacmanConf(playground = pacman_map3,
                            ghost_aggressivities = [0.72, 0.62], #, 0.52, 0.42],
                            edible_ghosts_behavior = [EdibleGhostBehaviors.FLEE_SLOW, EdibleGhostBehaviors.FLEE_SLOW],
                            n_ghosts = 2,
                            n_power_cookies = 2,
                            max_power_credit = 12,
                            eat_cookie_reward = 0.6,
                            eat_power_cookie_reward = 1.3,
                            eat_ghost_reward = 22,
                            living_cost = -0.15,
                            lose_reward = -35,
                            win_reward = 30,
                            penalty_per_ghost = -5)

env = PacmanEnvironment(pacman_conf)#, start_ghosts_pos=[(5,6), (5,8)])

agent = A2C_PacmanAgent([env], grayscale=False, n_frames=1)
#agent.load("./Trainings/A2C/2/pacman_color_a2c_env21_steps19_flee_slow_195000.pth")
agent.load("./Trainings/A2C/1/pacman_color_a2c_env21_steps19_185000.pth")

stockastic = True # When there is no ghost the action is stochastic (else argmax)

# agent = DQN_PacmanAgent([env], grayscale=False, n_frames=1)
# agent.load("pacman_color_dqn_tmp.pth")
# #agent.load("pacman_color_dqn_215000.pth")
# stockastic = False
  
for i in range(n_games):
    print(f'\r{i+1}', end='\r', flush=True)
    has_lost_game, aborted, n_ghost, n_eaten_power_cookies = play_game(env, agent, stockastic, display)
    if aborted == False:
        lost.append(has_lost_game)
    ghost_count[n_ghost] += 1.
    abort_count[n_ghost] += int(aborted)
    eaten_power_cookies_count[n_eaten_power_cookies] += 1.

print("",end='\n')

if n_games > 1:
    winnings = (1. - np.sum(lost)/len(lost))*100.0
    total_ghosts = 0.
    total_eaten_cookies = 0.
    
    print(f"Pacman won {winnings:.2f}% of the games")
    print("Number of game aborted:") 
    for k, v in dict(sorted(abort_count.items())).items():
        print(f"\twith {k} ghost: {v}")
    print("Percentage of games with")
    for k, v in dict(sorted(ghost_count.items())).items():
        print(f"\t{k} ghost: {(v *100) / n_games}")
        total_ghosts += k * v
    print(f"Average number of ghost per game: {total_ghosts/float(n_games)}")
    print("Percentage of power cookies eaten per game:")
    for k, v in dict(sorted(eaten_power_cookies_count.items())).items():
        print(f"\t{k} power cookies: {(v *100) / float(n_games)}")
        total_eaten_cookies += k * v
    print(f"Average number of power cookies eaten per game: {total_eaten_cookies/float(n_games)}")