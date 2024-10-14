#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:53:00 2024

@author: chris
"""

from pacman_conf import Pos
from pacman_maps import PacmanMap
import numpy as np
from typing import List
from dataclasses import dataclass
import heapq

from enum import Enum

class EdibleGhostBehaviors(Enum):
    FLEE_SLOW = 1
    FLEE_FAST = 2
    FREEZE = 3
    RANDOM = 4
    KAMIKAZE = 5

class PacmanGhost:
    def __init__(self, pacman_map: PacmanMap, pos: Pos, aggressiveness = 0.85, color="red",
                 id=0, heading:str=None, edible_ghost_behavior:EdibleGhostBehaviors=EdibleGhostBehaviors.FLEE_SLOW):
        self.map = pacman_map
        self.aggressiveness = aggressiveness
        self.current_aggressiveness = self.aggressiveness # Level of aggressiveness (when 0 the ghost must run away from pacman)
        self.pos = pos # Current position
        self.heading = heading # Current heading if any
        self.color = color
        self.id = id
        self.age = 0 # At each step of the game the ghost age is increased by 1
        self.edible_ghost_behavior = edible_ghost_behavior

    # Define the Node class
    @dataclass
    class Node:
        parent_pos: Pos
        f: float # Total cost of the node (g + h)
        g: float # Cost from start to this node
        h: float # Heuristic cost from this node to destination
            
    def _find_path_to(self, env) -> List[Pos]:
        dest: Pos  = env.current_pacman_pos
        # Implement the A* search algorithm
        def a_star_search(dest):
            path=[]
            
            # Check if the destination is reachable
            if self.map.is_blocked(*dest):
                return path
            
            # Check if we are already at the destination
            if self.pos == dest:
                return path
            
            # Initialize the closed list (visited nodes)
            closed_list = [[False for _ in range(self.map.columns)] for _ in range(self.map.rows)]
            # Initialize each nodes
            nodes = [[self.Node(None, float('inf'),float('inf'), 0)
                             for _ in range(self.map.columns)] for _ in range(self.map.rows)]
            
            # Initialize the start node
            i = self.pos[0]
            j = self.pos[1]
            nodes[i][j].f = 0.
            nodes[i][j].g = 0.
            nodes[i][j].h = 0.
            nodes[i][j].parent_i = i
            nodes[i][j].parent_j = j
            
            # Initialize the open list (nodes to be visited) with the start node
            open_list = []
            heapq.heappush(open_list, (0.0, i, j))
            
            # Flag that tells whether destination is found
            found_dest = False
            
            while len(open_list) > 0:
                # Pop the node with the smallest f value from the open list
                _, i, j = heapq.heappop(open_list)
            
                # Mark the node (i,j) as visited
                closed_list[i][j] = True
            
                # For each direction, check the successors 
                for d_i, d_j in env.dir.values():
                    new_i = i + d_i
                    new_j = j + d_j
                    
                    # If the successor is unblocked, and not visited
                    if self.map.is_unblocked(new_i, new_j) and not closed_list[new_i][new_j]:
                        # Check if we reach the destination
                        if (new_i, new_j) == dest:
                            # Set the parent of the destination cell
                            nodes[new_i][new_j].parent_i = i
                            nodes[new_i][new_j].parent_j = j
                            found_dest = True
                            break
                        else:
                            # Calculate the new f, g, and h values
                            g_new = nodes[i][j].g + 1.0
                            h_new = np.linalg.norm(np.array((new_i, new_j)) - np.array(dest))
                            f_new = g_new + h_new
                            
                            # If the cell is not in the open list or the new f value is smaller
                            if nodes[new_i][new_j].f == float('inf') or f_new < nodes[new_i][new_j].f:
                                # Add the cell to the open list
                                heapq.heappush(open_list, (f_new, new_i, new_j))
                                # Update the cell info
                                nodes[new_i][new_j].f = f_new
                                nodes[new_i][new_j].g = g_new
                                nodes[new_i][new_j].h = h_new
                                nodes[new_i][new_j].parent_i = i
                                nodes[new_i][new_j].parent_j = j
                        
                # If the destination is reached
                if found_dest:
                    row = dest[0]
                    col = dest[1]
                    
                    # Trace the path from destination to source using parent cells
                    while not (nodes[row][col].parent_i == row and nodes[row][col].parent_j == col):
                        path.append((row, col))
                        row, col = nodes[row][col].parent_i, nodes[row][col].parent_j
                    
                    # Add the source cell to the path
                    path.append(self.pos)
                    # Reverse the path to get the path from source to destination
                    path.reverse()
                    break
                    
            return path
                
        return a_star_search(dest)

    def move(self, env):
        ghost_pos = self.pos
        heading =  self.heading
        all_moves = np.asarray(list(set(map(tuple, env.map_actions[ghost_pos[0],ghost_pos[1]]))))
        # First, update the ghost aggressiveness level
        # (it will be 0 if pacman has power credit)
        aggressiveness = (env.power_credit == 0) *  self.aggressiveness
        if np.random.binomial(1, aggressiveness):
            # Attack pacman
            if self.id == 0 or self.id == 1:
                path = self._find_path_to(env)
                path_len = len(path)
                if path_len > 1:
                    new_pos = path[1]
                else:
                    new_pos = ghost_pos
            else:
                # Compute a probability weight for each possible move
                weights = np.ones(all_moves.shape[0], dtype=float) * env.map.max_dist
                for idx, pos in enumerate(all_moves):
                    if np.linalg.norm(ghost_pos - pos) == 0:
                        weights[idx] = 0
                        continue 
                    if heading != None and tuple(pos) == heading:
                        # Encourage to keep same direction
                        weights[idx] *= 1.5
                    # Go towards pacman
                    dist = np.linalg.norm(np.asarray(env.current_pacman_pos) - np.asarray(pos))
                    dist += 0.0001 # Avoid divide by 0
                    weights[idx] /= dist**2 # Square the distance to increase the differences

                p = weights / np.sum(weights)
                idx = np.argmax(np.random.multinomial(1, p))
                new_pos = all_moves[idx]
        else:
            weights = np.ones(all_moves.shape[0])
            for idx, pos in enumerate(all_moves):
                
                if aggressiveness == 0:
                    # pacman has the power to eat the ghosts
                    if self.edible_ghost_behavior == EdibleGhostBehaviors.FREEZE:
                        # The ghost freeze to its current position
                        if tuple(pos) == ghost_pos:
                            weights[idx] = 1
                        else:
                            weights[idx] = 0
                    else: 
                        dist = np.linalg.norm(np.asarray(env.current_pacman_pos) - np.asarray(pos))
                        if self.edible_ghost_behavior == EdibleGhostBehaviors.FLEE_SLOW:   
                            # Flee away from pacman
                            weights[idx] *= np.sqrt(dist) # use sqrt to give a chance to pacman to catch the ghost
                        elif self.edible_ghost_behavior == EdibleGhostBehaviors.FLEE_FAST:   
                            # Flee away from pacman
                            weights[idx] *= dist
                            if tuple(pos) == env.current_pacman_pos:
                                weights[idx] = 0
                        elif self.edible_ghost_behavior == EdibleGhostBehaviors.KAMIKAZE:  
                            dist += 0.0001 # In case dist == 0
                            weights[idx] *= 1./(dist**2)
                        else: # edible_ghost_behavior == RANDOM
                            # uniform distribution among the moves that do not 
                            # send the ghost into pacman.
                            if tuple(pos) == env.current_pacman_pos:
                                weights[idx] = 0
                            else:
                                weights[idx] = 1
                            
                elif tuple(pos) == heading:
                    # Encourage to keep same direction
                    weights[idx] *= 1.5
                    
            p = weights / np.sum(weights)
            idx = np.argmax(np.random.multinomial(1, p))
            new_pos = all_moves[idx]

        self.heading = env.heading(np.asarray(ghost_pos), new_pos)
        self.pos = tuple(new_pos)