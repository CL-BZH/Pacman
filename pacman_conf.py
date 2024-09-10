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
    ghost_aggressivities: List     # Defines how much the ghost are eager to attack pacman
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
    win_penalty_per_ghost: float   # Penalty for each ghost not eaten when winning the gmae

