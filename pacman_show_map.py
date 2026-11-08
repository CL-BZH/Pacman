#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show the board of a game version (or any maze of pacman_maps.py) and print a
few facts about the maze.

Examples:
    python pacman_show_map.py                     # game v2, a random start
    python pacman_show_map.py -g v1 --seed 3      # game v1, a reproducible start
    python pacman_show_map.py --map pacman_map2   # another maze, rules of v2
    python pacman_show_map.py --empty             # the maze alone
    python pacman_show_map.py -o board_v2.png     # save instead of showing
"""

import argparse
import dataclasses
import random
import sys

import numpy as np

import pacman_maps
from pacman_maps import PacmanMap
from pacman_env import PacmanEnvironment
from pacman_training import GAMES


def maze_facts(maze: PacmanMap):
    '''Size, free cells, dead ends and left-right symmetry of a maze.'''
    grid = maze.map
    dead_ends = 0
    for (i, j) in maze.rooms:
        neighbours = sum(maze.is_unblocked(i + di, j + dj)
                         for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                         if 0 <= i + di < maze.rows and 0 <= j + dj < maze.columns)
        dead_ends += neighbours == 1
    return {"size": f"{maze.rows} x {maze.columns}",
            "free cells": maze.n_rooms,
            "dead ends": dead_ends,
            "left-right symmetric": bool(np.array_equal(grid, grid[:, ::-1]))}


def main():
    maps = sorted(name for name in dir(pacman_maps)
                  if isinstance(getattr(pacman_maps, name), PacmanMap))
    parser = argparse.ArgumentParser(
        description="Show the board of a game version, or any maze of pacman_maps.py.")
    parser.add_argument("-g", "--game", choices=list(GAMES), default="v2",
                        help="game version (default: v2)")
    parser.add_argument("--map", choices=maps, default=None,
                        help="use this maze instead of the game's own, with the game's rules")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed: the start positions are random, the seed makes them reproducible")
    parser.add_argument("--empty", action="store_true",
                        help="show the maze alone (walls and free cells), without pieces")
    parser.add_argument("-o", "--output", default=None,
                        help="save the image to this file (e.g. board.png) instead of opening a window")
    args = parser.parse_args()

    import matplotlib
    if args.output:
        matplotlib.use("Agg")  # no window needed
    import matplotlib.pyplot as plt

    conf = GAMES[args.game]
    if args.map:
        conf = dataclasses.replace(conf, playground=getattr(pacman_maps, args.map))
    maze = conf.playground

    name = args.map or next(n for n in maps if getattr(pacman_maps, n) is maze)
    print(f"Maze {name} (game {args.game}):")
    for key, value in maze_facts(maze).items():
        print(f"  {key}: {value}")
    if not args.empty:
        print(f"  ghosts: {conf.n_ghosts}, power cookies: {conf.n_power_cookies}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    m = max(maze.rows, maze.columns)
    width, height = 5 * maze.columns / m, 5 * maze.rows / m
    if args.empty:
        fig, ax = plt.subplots(figsize=(width, height))
        maze.show(ax=ax)
        ax.get_legend().remove()
    else:
        try:
            env = PacmanEnvironment(conf)
            env.reset()
        except Exception as e:  # e.g. too many ghosts or power cookies for a small maze
            sys.exit(f"Cannot set up the game on this maze: {e}")
        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)
        ax.set_axis_off()
        env.colormap(width, height, fig, ax, 0.01)

    if args.output:
        fig.savefig(args.output, dpi=100, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
