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
    def __init__(self, pacman_map: PacmanMap, pos: Pos, aggressivity = 0.85, color="red",
                 id=0, heading:str=None, edible_ghost_behavior:EdibleGhostBehaviors=EdibleGhostBehaviors.FLEE_SLOW):
        self.map = pacman_map
        self.aggressivity = aggressivity
        self.current_aggressivity = self.aggressivity # Level of aggressivity (when 0 the ghost must run away from pacman)
        self.pos = pos # Current position
        self.heading = heading # Current heading if any
        self.color = color
        self.id = id
        self.age = 0 # At each step of the game the ghost age is increased by 1
        self.dir = {'E': (0,1), 'N': (-1,0), 'W': (0,-1), 'S': (1,0)}
        #print(f"Ghost {self.id} at pos {self.pos}")
        self.edible_ghost_behavior = edible_ghost_behavior

    # Define the Node class
    @dataclass
    class Node:
        parent_pos: Pos
        f: float # Total cost of the node (g + h)
        g: float # Cost from start to this node
        h: float # Heuristic cost from this node to destination
            
    def find_path_to(self, dest: Pos) -> List[Pos]:
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
                for d_i, d_j in self.dir.values():
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
