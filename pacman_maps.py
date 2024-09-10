#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:55:48 2024

@author: chris
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from typing import List

class PacmanMap:
    def __init__(self, ll: [List[List]]):
        self.map = np.array(ll)
        self.rows, self.columns = self.map.shape
        self.max_dist = np.sqrt(self.rows**2  + self.columns**2)
        self.start = None
        self.walls = set() # All blocked cells
        self.rooms = set() # All unblocked cells (i.e. all cells that are not walls)
        self.cells = {"room": (0,"yellow"),
                      "wall": (1,"black")}
        for i in range(self.rows):
            for j in range(self.columns):
                if self.map[i,j] == 1:
                    self.walls.add((i,j))
                else:
                    self.rooms.add((i,j))                    
    
    def __getitem__(self, item):
        i = item[0]
        j = item[1]
        assert i>=0 and i<self.rows
        assert j>=0 and j<self.columns
        return self.map[i, j]
    
    def __setitem__(self, idx, val):
        pass
        
    def is_unblocked(self, i, j):
    	return self.map[i,j] == 0
        
    def is_blocked(self, i, j):
    	return self.map[i,j] == 1

    @property
    def n_rooms(self):
        return len(self.rooms)
    
    def show(self, pos_list=[], ax=None):
        map = self.map.copy()
        
        if ax == None:
            m = 0.9*max(self.rows, self.columns)
            width = 3 * float(self.columns)/m
            height = 3 * float(self.rows)/m
            fig, ax = plt.subplots(1,1,figsize=(width, height))
        #ax.grid( which='major')
        ax.set_axis_off()

        colours = self.cells

        colors = [c for (_,c) in colours.values()]

        cell_types = list(colours.keys())
        cmap = ListedColormap(colors)
        Z = map[::-1]
        ax.pcolormesh(Z, cmap=cmap, edgecolors='k', alpha=0.8)
        patches = [mpatches.Patch(color=colors[i], label="{t}".format(t=cell_types[i])) for i in range(len(colors))]
        if pos_list != []:
            for i, pos in enumerate(pos_list):
                x = pos[1] + 0.5
                y = self.rows - pos[0] - 0.5
                if i == 0:
                    circle = mpatches.Circle((x, y), 0.25, color="blue", ec="none", label="Start {t}".format(t=pos))
                    patches.append(circle)
                elif i == len(pos_list) - 1:
                    circle = mpatches.Circle((x, y), 0.25, color="red", ec="none", label="End {t}".format(t=pos))
                    patches.append(circle)
                else:
                    circle = mpatches.Circle((x, y), 0.25, color="silver", ec="none")
                ax.add_artist(circle)
            
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )




pacman_map1 = PacmanMap([[1,1,1,1,1,1,1],
                        [1,0,0,0,0,0,1],
                        [1,0,1,0,1,0,1],
                        [1,0,1,0,1,0,1],
                        [1,0,0,0,0,0,1],
                        [1,1,1,1,1,1,1]])

pacman_map2 = PacmanMap([ [1,1,1,1,1,1,1,1,1,1],
                         [1,0,0,0,0,0,0,0,0,1],
                         [1,0,1,0,1,0,1,1,0,1],
                         [1,0,0,0,1,0,0,0,0,1],
                         [1,0,1,0,0,0,1,1,0,1],
                         [1,0,0,0,1,0,0,0,0,1],
                         [1,1,1,1,1,1,1,1,1,1]])

pacman_map3 = PacmanMap([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                         [1,0,1,0,1,1,0,1,0,1,1,0,1,0,1],
                         [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
                         [1,0,1,1,0,1,1,1,1,1,0,1,1,0,1],
                         [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                         [1,1,1,0,1,1,0,1,0,1,1,0,1,1,1],
                         [1,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
                         [1,0,1,0,1,0,1,1,1,0,1,0,1,0,1],
                         [1,0,0,0,0,0,1,1,1,0,0,0,0,0,1],
                         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])