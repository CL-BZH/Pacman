#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:24:25 2024

@author: chris
"""

"""
Let a trained agent play Pacman.

Examples:
    python pacman_play.py                       # display one game of ./Tmp/pacman_a2c_final.pth
    python pacman_play.py -n 1000               # statistics over 1000 games
    python pacman_play.py -a dqn -c my_dqn.pth  # DQN agent with another checkpoint
    python pacman_play.py --help                # all the options
"""

from pacman_training import pacman_conf  # same game configuration as for the training
from pacman_env import PacmanEnvironment
from pacman_agents import DQN_PacmanAgent
from pacman_agents import A2C_PacmanAgent
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
import argparse
import os
import sys
import time
from collections import defaultdict


def play_game(env, agent, stockastic=False, display=False, wait=15, speed=0.12, max_moves=800):
    '''
    Play one game.
        stockastic: if True, the action is sampled from the policy once all ghosts are eaten
                    (the greedy action is always used while ghosts are present).
        display: show the game in a window.
        wait: seconds to wait before the game starts (display only).
        speed: pause between two moves in seconds (display only).
        max_moves: the game is aborted after this number of moves.
    Returns (lost, aborted, number of ghosts left, number of power cookies eaten).
    '''

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
        env.colormap(width, height, fig, ax, speed)
        plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 
        plt.pause(max(wait, 0.01))
        ax.cla()
        
    rewards = []
    aborted = False
    
    state = agent.current_state[0]
    while env.end == False:
        if env.ghosts != []:
            _, action = agent.select_action(state, stockastic=False)
        else:
            _, action = agent.select_action(state, stockastic=stockastic)

        reward = env.step(action)
        state = agent.observe(0)
        agent.episode_rewards.append(reward)
        rewards.append(reward)
        
        if display == True:
            env.colormap(width, height, fig, ax, speed)
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
                env.colormap(width, height, fig, ax, speed)
                time.sleep(0.7)
                plt.close();
        if len(rewards) > max_moves:
            print(f"\nAbort.Number of ghost: {len(env.ghosts)}.") #"\nRewards:\n{rewards}")
            aborted = True
            break
        
    # Return True if Pacman lost the game
    #print(f"Last reward: {rewards[-1]}")
    return env.dead == True, aborted, len(env.ghosts), env.n_eaten_power_cookies


def parse_args():
    parser = argparse.ArgumentParser(
        description="Let a trained agent play Pacman. With one game the game is displayed; "
                    "with several games, statistics are printed at the end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--agent", choices=["a2c", "dqn"], default="a2c",
                        help="type of agent")
    parser.add_argument("-c", "--checkpoint", default=None,
                        help="network parameters (.pth); default: ./Tmp/pacman_<agent>_final.pth")
    parser.add_argument("-n", "--games", type=int, default=1,
                        help="number of games to play")
    parser.add_argument("-f", "--frames", type=int, default=2,
                        help="number of stacked frames (must match the training)")
    parser.add_argument("--greedy", action="store_true",
                        help="always play the best action (by default the A2C agent samples its "
                             "action from the policy once all ghosts are eaten)")
    parser.add_argument("--display", action="store_true",
                        help="display the games even when several games are played")
    parser.add_argument("--wait", type=float, default=3.,
                        help="seconds to wait before a displayed game starts")
    parser.add_argument("--speed", type=float, default=0.12,
                        help="pause between two moves of a displayed game (seconds)")
    parser.add_argument("--max-moves", type=int, default=800,
                        help="games longer than this are aborted")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (to replay the same games)")
    return parser.parse_args()


def print_statistics(n_games, lost, abort_count, ghost_count, eaten_power_cookies_count):
    if lost:
        winnings = (1. - np.sum(lost)/len(lost))*100.0
        print(f"Pacman won {winnings:.2f}% of the games")
    else:
        print("All the games were aborted")
    total_ghosts = 0.
    total_eaten_cookies = 0.
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


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    checkpoint = args.checkpoint or f"./Tmp/pacman_{args.agent}_final.pth"
    if not os.path.isfile(checkpoint):
        sys.exit(f"Checkpoint not found: {checkpoint}")

    # Same game configuration as for the training
    env = PacmanEnvironment(pacman_conf)

    if args.agent == "a2c":
        agent = A2C_PacmanAgent([env], n_frames=args.frames)
        stockastic = not args.greedy
    else:
        agent = DQN_PacmanAgent([env], n_frames=args.frames)
        agent.epsilon = 0.
        stockastic = False
    agent.load(checkpoint)
    print(f"Loaded {checkpoint}")

    display = args.display or args.games == 1
    lost = []
    abort_count = defaultdict(int)
    ghost_count = defaultdict(int)
    eaten_power_cookies_count = defaultdict(int)

    for i in range(args.games):
        print(f'\r{i+1}', end='\r', flush=True)
        has_lost_game, aborted, n_ghost, n_eaten_power_cookies = play_game(
            env, agent, stockastic, display, wait=args.wait, speed=args.speed, max_moves=args.max_moves)
        if aborted == False:
            lost.append(has_lost_game)
        ghost_count[n_ghost] += 1.
        abort_count[n_ghost] += int(aborted)
        eaten_power_cookies_count[n_eaten_power_cookies] += 1.

    print("",end='\n')

    if args.games > 1:
        print_statistics(args.games, lost, abort_count, ghost_count, eaten_power_cookies_count)
    else:
        result = "aborted" if abort_count[n_ghost] else ("lost" if has_lost_game else "won")
        print(f"Game {result}. Ghosts left: {n_ghost}. Power cookies eaten: {n_eaten_power_cookies}.")


if __name__ == "__main__":
    main()
