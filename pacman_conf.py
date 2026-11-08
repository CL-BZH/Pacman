#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:57:24 2024

@author: chris
"""

from pacman_maps import PacmanMap
from dataclasses import dataclass
from typing import Tuple, List, ClassVar
from typing import TypeAlias

Pos: TypeAlias = Tuple[int, int]

@dataclass
class PacmanConf:
    max_n_ghost: ClassVar[int] = 4 # Maximum number of ghost
    playground: PacmanMap          # The playground map
    ghost_aggressiveness_levels: List     # Defines how much the ghost are eager to attack pacman
    edible_ghosts_behavior: List   # Behavior of each ghost when it becomes edible
    n_ghosts: int                  # Number of ghost at reset
    n_power_cookies: int           # Number of power cookies (i.e. cookies that give the power to pacman to eat ghost)
    max_power_credit: int          # Duration of power credit for eating ghost
    eat_cookie_reward: float       # Reward for eating a cookie
    eat_power_cookie_reward: float # Reward for eating a power cookie
    eat_ghost_reward: float        # Reward for eating a ghost
    living_cost:float              # Reward for living
    lose_reward: float             # Reward for being eaten by a ghost
    win_reward: float              # Reward for finishing the game (ate all cookies) without being eaten by a ghost
    penalty_per_ghost: float       # Penalty for each ghost not eaten when winning the gmae
    ghost_respawn_delay: int = 0   # An eaten ghost comes back after this number of moves, and not before
                                   # Pacman's power has ended (0: eaten ghosts never come back)
    ghost_respawn_min_distance: int = 6 # A ghost comes back at a random cell at least this number of
                                        # moves away from Pacman
    ghost_keep_direction: bool = False  # When a ghost moves randomly, the move that keeps its direction
                                        # has a weight 1.5 (False in game v1, where the intended
                                        # preference was never applied because of a bug)

