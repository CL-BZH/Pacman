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
    '''
    A board: 0 = free cell, 1 = wall, 2 = ghost house (optional).
    The ghost house is blocked for Pacman (and for the moving ghosts): ghosts wait in it and
    leave it through its door, a free cell next to it (by default the free cell above the
    middle of the house; it can be given with house_door).
    '''
    def __init__(self, ll: [List[List]], house_door=None):
        self.map = np.array(ll)
        self.rows, self.columns = self.map.shape
        self.max_dist = np.sqrt(self.rows**2  + self.columns**2)
        self.start = None
        self.walls = set() # All blocked cells (the ghost house included)
        self.rooms = set() # All unblocked cells (i.e. all cells that are not walls)
        self.cells = {"room": (0,"yellow"),
                      "wall": (1,"black")}
        for i in range(self.rows):
            for j in range(self.columns):
                if self.map[i,j] != 0:
                    self.walls.add((i,j))
                else:
                    self.rooms.add((i,j))
        # Ghost house: its cells (one per ghost, ghost i waits in house[i % len(house)]) and its door
        self.house = sorted((i, j) for i in range(self.rows) for j in range(self.columns)
                            if self.map[i, j] == 2)
        self.house_door = None
        if self.house:
            self.cells["house"] = (2, "#9b7bc4")
            if house_door is None:
                rows = [i for i, _ in self.house]
                cols = [j for _, j in self.house]
                centre = (min(cols) + max(cols)) / 2
                candidates = [(i - 1, j) for i, j in self.house if (i - 1, j) in self.rooms]
                candidates += [(i + 1, j) for i, j in self.house if (i + 1, j) in self.rooms]
                assert candidates, "the ghost house needs a free cell above or below it (its door)"
                house_door = min(candidates, key=lambda c: (abs(c[1] - centre), c[0] > min(rows)))
            assert tuple(house_door) in self.rooms, "the door of the ghost house must be a free cell"
            self.house_door = tuple(house_door)
    
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


# Bigger maze (17 x 19, 163 open cells): left-right symmetric, no dead ends.
# Ghost house (2) of 3 cells in the middle, with its door above it at (7, 9).
pacman_map4 = PacmanMap([
                          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                          [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                          [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
                          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                          [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
                          [1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
                          [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
                          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                          [1,1,1,1,0,1,0,1,2,2,2,1,0,1,0,1,1,1,1],
                          [1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
                          [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
                          [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
                          [1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1],
                          [1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
                          [1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1],
                          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
