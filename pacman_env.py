#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:04:33 2024

@author: chris
"""

from pacman_conf import PacmanConf, Pos
from ghost_agents import PacmanGhost, EdibleGhostBehaviors
import random
import numpy as np
import copy
#import time
from IPython.display import display, clear_output
import numpy.matlib
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from typing import Optional, List



class PacmanEnvironment:
    '''
    '''
    def __init__(self, config: PacmanConf, start_pacman_pos: Optional[Pos]=None, start_ghosts_pos: Optional[List[Pos]]=None):
        # Apply the config
        self.map = copy.deepcopy(config.playground)
        self.n_ghosts = config.n_ghosts
        self.ghost_aggressivities = config.ghost_aggressivities
        self.edible_ghosts_behavior = config.edible_ghosts_behavior
        self.n_power_cookies = config.n_power_cookies
        self.n_eaten_power_cookies = 0
        self.max_power_credit = config.max_power_credit
        self.eat_cookie_reward = config.eat_cookie_reward
        self.eat_power_cookie_reward = config.eat_power_cookie_reward
        self.eat_ghost_reward = config.eat_ghost_reward
        self.living_cost = config.living_cost
        self.lose_reward = config.lose_reward 
        self.win_reward = config.win_reward
        self.win_penalty_per_ghost = config.win_penalty_per_ghost
        self.n_steps = 0
        self.end = False
        self.dead = False
        self.pacman_heading = None
        self.power_credit = 0 # Start with no power for eating ghost
        self.ghost_colors = ["red", "magenta", "darkorange", "pink"]
        self.actions = {'E': (0, (0,1)), #'East' is at index 0 and change position: i->i+0,  j->j+1
                        'N': (1, (-1,0)),
                        'W': (2, (0,-1)),
                        'S': (3, (1,0)),
                        'I': (4, (0, 0))} #'Idle' is at index 4 and change position: i->i+0,  j->j+0
        self.headings = {(0,1): 'E',
                         (-1,0): 'N',
                         (0,-1): 'W',
                         (1,0): 'S',
                         (0,0): None}
        self.invalid_action = -1
        self.game_over = False
        # Register the valid positions on the map
        self.valid_pos = list(self.map.rooms)
        self.cookies = []
        self.current_pacman_pos = None
        self.start_pacman_pos = None
        self.start_ghosts_pos = start_ghosts_pos
        self.visited_pos = None
        self.ghosts = None
        self.dead_ghosts = None
        self.last_eaten_cookie = {'pos': (0,0), 'power': False, 'steps': 0}
        self.power_cookies = None # Cookies that give power to eat ghost
        self.fill_map(start_pacman_pos, start_ghosts_pos)
        self.map_actions = np.array(self._init_map_actions())

        
    def reset(self, pacman_start_pos: Optional[Pos]=None, ghosts_start_pos: Optional[List[Pos]]=None):
        self.n_steps = 0
        self.end = False
        self.dead = False
        self.current_pacman_pos = None
        self.start_pacman_pos = None
        self.visited_pos = None
        self.pacman_heading = None
        self.power_credit = 0
        self.game_over = False
        self.n_eaten_power_cookies = 0
        self.last_eaten_cookie = {'pos': (0,0), 'power': False, 'steps': 0}
        if ghosts_start_pos == None:
            ghosts_start_pos = self.start_ghosts_pos
        self.fill_map(pacman_start_pos, ghosts_start_pos)
        
    def fill_map(self, pacman_start_pos: Optional[Pos]=None, ghosts_start_pos: Optional[List[Pos]]=None):
        
        def add_cookies():
            self.cookies = list(self.map.rooms)
    
        def add_pacman(start_pacman_pos, start_ghosts_pos):
            #assert self.cookies != []
            while start_pacman_pos == None:
                start_pacman_pos = random.sample(list(self.valid_pos), 1)[0]
                if ghosts_start_pos != None and np.min(cdist([start_pacman_pos], ghosts_start_pos)) < 2:
                    start_pacman_pos = None
            #assert start_pacman_pos in self.valid_pos
            self.current_pacman_pos = start_pacman_pos
            self.start_pacman_pos = start_pacman_pos
            self.visited_pos = set([start_pacman_pos])
            self.cookies.remove(start_pacman_pos) # Remove the cookie where pacman is
            
        def add_power_cookies_and_ghost(start_ghosts_pos):
            #assert self.start_pacman_pos != None
            # Place ghosts
            self.ghosts = []
            self.dead_ghosts = []
            ghosts_pos = []
            if start_ghosts_pos == None:
                while ghosts_pos == []:
                    ghosts_pos = random.sample(self.cookies, self.n_ghosts)
                    if np.min(cdist([self.start_pacman_pos], ghosts_pos)) < 2:
                        ghosts_pos = []    
            else:
                ghosts_pos = start_ghosts_pos
            for i, pos in enumerate(ghosts_pos):
                color = self.ghost_colors[i]
                aggressivity = self.ghost_aggressivities[i]
                edible_ghost_behavior = self.edible_ghosts_behavior[i]
                self.ghosts.append(PacmanGhost(self.map, pos, aggressivity,
                                               id=i, color=color,
                                               edible_ghost_behavior=edible_ghost_behavior))
            # Place power cookies (replace normal cookies by power cookies)
            self.power_cookies = random.sample(self.cookies, 1) # Place the first power cookie
            self.cookies.remove(self.power_cookies[-1]) # "normal cookies" are replaced by "power cookies"
            n_power_cookies =  self.n_power_cookies - 1
            while n_power_cookies != 0:
                dist = cdist(self.power_cookies, self.cookies)
                dmin = np.min(dist, axis=0)
                farest_pos = self.cookies[np.argmax(dmin)]
                dist = cdist([farest_pos], self.cookies)
                dmax = np.max(dist)
                dist += dmax
                w = 1.0/dist
                p = w/np.sum(w)
                power_cookie_pos = random.choices(self.cookies, k=1, weights=p[0])[0]
                self.power_cookies.append(power_cookie_pos)
                self.cookies.remove(power_cookie_pos)
                n_power_cookies -= 1

        # Place all objects on the map. order is important:
        # 1st: all cookies
        # 2nd: Pacman
        # 3rd: Ghost and power cookies
        add_cookies()
        add_pacman(pacman_start_pos, ghosts_start_pos)
        add_power_cookies_and_ghost(ghosts_start_pos)
    
    def _eat_cookie(self):
        '''
        If at the position 'pos' there is a cookie (with power or not) eat it.
        '''
        pos = self.current_pacman_pos
        reward = 0

        if pos in self.cookies:
            self.last_eaten_cookie = {'pos': pos, 'power': False, 'steps': self.n_steps}
            self.cookies.remove(pos)
            if self.power_credit == 0 or len(self.ghosts) == 0:
                # Eating a cookie gives a positive reward only if
                # Pacman has no power or if he has power but there is no more ghost to eat
                reward = self.eat_cookie_reward
        elif pos in self.power_cookies:
            self.last_eaten_cookie = {'pos': pos, 'power': True, 'steps': self.n_steps}
            self.power_cookies.remove(pos)
            self.n_eaten_power_cookies += 1
            if len(self.ghosts) == 0:
                # When there is no giost left then eating a power cookie is
                # the same as eating a normal cookie
                reward = self.eat_cookie_reward
            else:
                # The benefit of eating a power cookie decreases wrt the
                # current power credit pacman owns
                benefit = 1.0 - (float(self.power_credit) / float(self.max_power_credit))
                # The benefit of eating a power cookie is bigger the closer a ghost is
                min_dist = np.inf
                for ghost in self.ghosts:
                    dist = cdist(np.asarray(self.current_pacman_pos).reshape((1,2)),
                                 np.asarray(ghost.pos).reshape((1,2)),
                                 metric='cityblock')[0][0]
                    if dist < 1.0:
                        print(f"BUG!!! Ghost ID: {ghost.id}. Ghost pos: {ghost.pos}. Pacman pos: {self.current_pacman_pos}. Power: {self.power_credit}. Steps: {self.n_steps}. dist: {dist}")
                        dist = 1.0
                    if dist < min_dist:
                        min_dist = dist
                    #assert min_dist >= 1.0
                benefit *= 3.5/np.sqrt(min_dist)
                if min_dist > self.max_power_credit:
                    # Ghosts are too far, power cookie should not be eaten
                    benefit = 0.
                reward = self.eat_power_cookie_reward * benefit
                #print(f"min_dist: {min_dist}, benefit: {benefit}, reward: {reward}")
            self.power_credit = self.max_power_credit # Gain power credit for eating ghost
        if len(self.cookies) + len(self.power_cookies) == 0:
            reward += self.win_reward + (self.win_penalty_per_ghost * len(self.ghosts))
            self.end = True
        return reward
        
    def _check_collision(self, ghost):
        reward = 0
        if self.current_pacman_pos == ghost.pos:
            # Collision between pacman and a ghost
            if self.power_credit != 0:
                # Pacman eats the ghost
                self.ghosts.remove(ghost)
                self.dead_ghosts.append(ghost)
                reward = self.eat_ghost_reward
            else:
                # The ghost eats pacman
                self.dead = True
                self.end = True
                reward = self.lose_reward
        return reward
        
    def _eat_ghost(self):
        reward = 0
        ghosts = [ghost for ghost in self.ghosts]
        for ghost in ghosts:
            reward += self._check_collision(ghost)
            if self.dead:
                # The ghost eats pacman
                break
        return reward       
    
    def _init_map_actions(self):
        '''
        Build a map, where the 2 first dimensions stores the valid
        position and the 3rd dimension stores the position after
        each of the different actions.
        '''
        map_actions = []
        
        def valid_neighbor(i, j):
            '''
            Return False if the neigbor cell is a wall
            '''
            if (i, j) in self.map.walls:
                return False
            return True

        for i in range(self.map.rows):
            map_actions.append([])
            for j in range(self.map.columns):
                map_actions[i].append([])
                for (idx, (d_i, d_j)) in self.actions.values():
                    map_actions[i][j].append([])
                    if (i, j) in self.valid_pos:
                        new_i, new_j = i+d_i, j+d_j
                        if valid_neighbor(new_i, new_j):
                            map_actions[i][j][idx] = (new_i, new_j)
                        else:
                            # Hit a wall => stay at the same position
                            map_actions[i][j][idx] = (i, j)
                    else:
                        map_actions[i][j][idx] = (-1, -1)
         
        return map_actions
            
    def _update_pacman_pos(self, action):
        '''
        return the new position once 'action' is performed in position 'pos'.
        e.g. Let's say pos = (1,1) and action is 'E'. Then if there is no wall
        on the east side the return position is (1,2). If there is a wall then
        the return position is (1,1).
        
        '''
        reward = 0
        if self.power_credit == 0 or len(self.ghosts) == 0:
            # Small cost of living (encourage to move and eat cookies)
            reward = self.living_cost
        
        (i,j) = self.current_pacman_pos
        
        #print(f"_update_pacman_pos({action})")
        action_idx = self.actions[action][0]
        new_pos = self.map_actions[i, j, action_idx]
        heading = self._heading(self.current_pacman_pos, new_pos)
        self.pacman_heading = heading
        new_pos = tuple(new_pos)
        self.visited_pos.add(new_pos)
        
        if ((self.power_credit != 0 or len(self.ghosts) == 0)
            and self.current_pacman_pos == new_pos):
                # Discourage pacman to stay still when he has the power to eat
                # ghosts or when there is no ghost left
                reward = self.living_cost * 5
            
        self.current_pacman_pos = new_pos
        self.power_credit = max(0, self.power_credit - 1) # Decrease the power credit by 1 at each move
        
        reward += self._eat_ghost() # Eat a ghost if has power otherwise die
        if self.dead == False:
            reward += self._eat_cookie() # eat a cookie is there is one
        return reward

    def _move_ghost(self, ghost):
        ghost_pos = ghost.pos
        heading = ghost.heading
        all_moves = np.asarray(list(set(map(tuple, self.map_actions[ghost_pos[0],ghost_pos[1]]))))
        # First, update the ghost aggressivity level
        # (it will be 0 if pacman has power credit)
        aggressivity = (self.power_credit == 0) * ghost.aggressivity
        if np.random.binomial(1, aggressivity):
            # Attack pacman
            if ghost.id == 0 or ghost.id == 1:
                path = ghost.find_path_to(self.current_pacman_pos)
                path_len = len(path)
                if path_len > 1:
                    new_pos = path[1]
                else:
                    # if ghost.id != 0:
                    #     print(f"path: {path}. Ghost ID: {ghost.id}. Ghost pos: {ghost.pos}. Pacman pos: {self.current_pacman_pos}. Steps: {self.n_steps}")
                    #     assert ghost.id == 0
                    new_pos = ghost_pos
            else:
                # Compute a probability weight for each possible move
                weights = np.ones(all_moves.shape[0], dtype=float) * self.map.max_dist
                for idx, pos in enumerate(all_moves):
                    if numpy.linalg.norm(ghost_pos - pos) == 0:
                        weights[idx] = 0
                        continue 
                    if heading != None and tuple(pos) == heading:
                        # Encourage to keep same direction
                        weights[idx] *= 1.5
                    # Go towards pacman
                    dist = numpy.linalg.norm(np.asarray(self.current_pacman_pos) - np.asarray(pos))
                    dist += 0.0001 # Avoid divide by 0
                    weights[idx] /= dist**2 # Square the distance to increase the differences

                p = weights / np.sum(weights)
                idx = np.argmax(np.random.multinomial(1, p))
                new_pos = all_moves[idx]
        else:
            weights = np.ones(all_moves.shape[0])
            for idx, pos in enumerate(all_moves):
                
                if aggressivity == 0:
                    # pacman has the power to eat the ghosts
                    if ghost.edible_ghost_behavior == EdibleGhostBehaviors.FREEZE:
                        # The ghost freeze to its current position
                        if tuple(pos) == ghost_pos:
                            weights[idx] = 1
                        else:
                            weights[idx] = 0
                    else: 
                        dist = numpy.linalg.norm(np.asarray(self.current_pacman_pos) - np.asarray(pos))
                        if ghost.edible_ghost_behavior == EdibleGhostBehaviors.FLEE_SLOW:   
                            # Flee away from pacman
                            weights[idx] *= np.sqrt(dist) # use sqrt to give a chance to pacman to catch the ghost
                        elif ghost.edible_ghost_behavior == EdibleGhostBehaviors.FLEE_FAST:   
                            # Flee away from pacman
                            weights[idx] *= dist
                            if tuple(pos) == self.current_pacman_pos:
                                weights[idx] = 0
                        elif ghost.edible_ghost_behavior == EdibleGhostBehaviors.KAMIKAZE:  
                            dist += 0.0001 # In case dist == 0
                            weights[idx] *= 1./(dist**2)
                        else: # edible_ghost_behavior == RANDOM
                            # uniform distribution among the moves that do not 
                            # send the ghost into pacman.
                            if tuple(pos) == self.current_pacman_pos:
                                weights[idx] = 0
                            else:
                                weights[idx] = 1
                            
                elif tuple(pos) == heading:
                    # Encourage to keep same direction
                    weights[idx] *= 1.5
                    
            p = weights / np.sum(weights)
            idx = np.argmax(np.random.multinomial(1, p))
            new_pos = all_moves[idx]

        ghost.heading = self._heading(np.asarray(ghost_pos), new_pos)
        ghost.pos = tuple(new_pos)
        reward = self._check_collision(ghost)
        return reward

    def _heading(self, current_pos, next_pos):
        h = tuple(next_pos - current_pos)
        return self.headings[h]
        
    def step(self, action):
        self.n_steps += 1
        reward = 0
        #assert self.end == False
        # First the ghost with id 1 is updated
        for ghost in self.ghosts:
            ghost.age += 1
            if ghost.id == 1:
                reward = self._move_ghost(ghost)
                if self.end == True:
                    return reward
                break
        # Then pacman move to a new position
        reward += self._update_pacman_pos(action)
        if self.end == True:
           return reward
        for ghost in self.ghosts:
            if ghost.id == 1:
                continue
            reward += self._move_ghost(ghost)
            if self.end == True:
                break
        return reward

    def get_all_actions(self):
        '''
        Returns the list of possible actions.
        ['E', 'N', 'W', 'S', 'I']
        '''
        return list(self.actions.keys())

    @property
    def n_actions(self):
        '''
        Returns the number of possible actions.
        '''
        return len(self.get_all_actions())
        
    @property
    def state(self):
        return (self.map_actions, # Map of the ground with actions possible in each position
                self.current_pacman_pos, # Pacman current position on the map
                self.pacman_heading, # Pacman current heading
                [ghost.pos for ghost in self.ghosts], # All ghosts positions on the map
                self.cookies, # Positions of normal cookies
                self.power_cookies, # Positions of power cookies
                self.power_credit, # Pacman's current power credit
                self.visited_pos) # Already eaten cookies (visited positions)

    @property
    def rgb_image_shape(self):
        rows = self.map_actions.shape[0]
        columns = self.map_actions.shape[1]
        # Borders are removed from the image (see function rgb_image() below).
        rows -= 2
        columns -= 2
        return (rows, columns, 3)
        
    @property
    def rgb_image(self):
        """
        Return the Pacman environment state as a RGB image 
        that can be used as input of a CNN.
        """
        #n_actions = self.map_actions.shape[2]
        
        ghosts_pos = [ghost.pos for ghost in self.ghosts]# All ghosts positions on the map
        
        # RGB image
        rows = self.map_actions.shape[0]
        columns = self.map_actions.shape[1]
        rgb = np.zeros((rows, columns, 3), dtype=np.float32)
        
        # Green channel for the ghosts' color
        g = float(self.power_credit) / self.max_power_credit

        for i in range(rows):
            for j in range(columns):
                if (i,j) == self.current_pacman_pos:
                    b = 1. - g/2
                    rgb[i,j] = [0,0,b] # Blue pacman 
                elif (i,j) in ghosts_pos:
                    rgb[i,j] = [1,g,0] # From red to pinkish ghost
                elif (i,j) in self.cookies:
                    rgb[i,j] = [0,1,0] # Green cookie
                elif (i,j) in self.power_cookies:
                    rgb[i,j] = [0,1,1] # Blueish cookie
                elif (i,j) in self.visited_pos:
                    rgb[i,j] = [1,1,1] # White cells

        # First and last rows are removed since they are the upper and lowerborders
        rgb_del = np.delete(rgb, [0, -1], axis=0)
        # First and last columns are removed since they are the left and right borders
        rgb_del = np.delete(rgb_del, [0, -1], axis=1)

        return rgb_del
        
    def colormap(self, width, height, fig, ax):
        
        # Get all ghosts positions and headings
        ghosts_pos = list()
        ghosts_heading = list()
        if self.ghosts != []:
            ghosts = np.array([(ghost.pos, ghost.heading, ghost.color) for ghost in self.ghosts], dtype=object)
            ghosts_pos = list(ghosts[:,0])
            ghosts_heading = list(ghosts[:,1])
            ghosts_color = list(ghosts[:,2])
        
        # Special for ghost that just died (eaten by Pacman)
        eaten_ghosts_pos = []
        for dead_ghost in self.dead_ghosts:
            if dead_ghost.age == self.n_steps:
                #if dead_ghost.pos != self.current_pacman_pos:
                #    print(f"Ghost {dead_ghost.id} current pos: {dead_ghost.pos}\nCurrent Pacman pos: {self.current_pacman_pos}")
                #    assert dead_ghost.pos == self.current_pacman_pos
                eaten_ghosts_pos.append(dead_ghost.pos)
                
                
        cells = {"wall": (0,"black")} # Walls
        cells["valid"] = (1,"silver") # Rooms (i.e not wall cells)

        pacman_normal_color = '#0000FF'
        pacman_normal_ec = None
        pacman_color = pacman_normal_color
        pacman_alpha = 1.
        pacman_ec = pacman_normal_ec
        power_ratio = 0.
        power_cookie_color = '#00FF00'
        cookie_color = "yellow"
        edible_ghost_color = "gold"
        just_ate_power_cookie = (self.last_eaten_cookie['steps'] == self.n_steps
                                 and self.last_eaten_cookie['power'] == True)
        
        rows = self.map.rows
        columns = self.map.columns

        power_cookies = []
        cookies = []
        ghosts = []
        eaten_ghosts = []
        Z = np.zeros([rows, columns]) # Start with all black (== walls)
        for i in range(rows):
            for j in range(columns):
                if (i,j) in self.valid_pos:
                    Z[i, j] = cells["valid"][0]
                else:
                    Z[i, j] = cells["wall"][0]
                if (self.cookies != [] and 
                    (i,j) in self.cookies):
                    x = j + 0.5
                    y = rows - i - 0.5
                    cookies.append((x,y))
                if (self.power_cookies != [] and 
                    (i,j) in self.power_cookies):
                    x = j + 0.5
                    y = rows - i - 0.5
                    power_cookies.append((x,y))
                if (ghosts_pos != [] and 
                    (i,j) in ghosts_pos):
                    ghosts_idx = [k for k, pos in enumerate(ghosts_pos) if pos == (i,j)]
                    x = j + 0.42
                    y = rows - i - 0.62
                    if len(ghosts_idx) > 1:
                        for idx in ghosts_idx:
                            color = ghosts_color[idx]
                            heading = ghosts_heading[idx]
                            # shift a bit
                            xs = x + (random.random() - 0.5) * 0.3
                            ys = y + (random.random() - 0.5) * 0.3
                            ghosts.append((xs, ys, color, heading))
                    else:
                        idx = ghosts_idx[0]
                        color = ghosts_color[idx]
                        heading = ghosts_heading[idx]
                        ghosts.append((x, y, color, heading))
                if (eaten_ghosts_pos != [] and 
                    (i,j) in eaten_ghosts_pos):
                    x = j + 0.42
                    y = rows - i - 0.62
                    eaten_ghosts = [(x,y) for pos in eaten_ghosts_pos if pos == (i,j)]   
                if (i,j) == self.current_pacman_pos:
                    pacman_x = j + 0.54
                    pacman_y = rows - i - 0.52


        pacman_heading = self.pacman_heading
        if pacman_heading == None:
            pacman_heading = 'N'
            
        # Pacman with open mouth in direction of its heading
        pacmans_open = {'E': ((0.3,), {'theta1':40, 'theta2':320}),
                        'W': ((0.3,), {'theta1':220, 'theta2':140}),
                        'S': ((0.3,), {'theta1':-50, 'theta2':230}),
                        'N': ((0.3,), {'theta1':130, 'theta2':50})}

        match pacman_heading:
            case 'E':
                pacman_open = pacmans_open['E']
            case 'S':
                pacman_open = pacmans_open['S']
            case 'W':
                pacman_open = pacmans_open['W']
            case 'N':
                pacman_open = pacmans_open['N']

        n_frames = 3
        for f in range(n_frames):
            # Clear the output once the new one is ready
            clear_output(wait=True)
        
            if self.power_credit != 0:
                pacman_color = power_cookie_color
                pacman_ec = "black"
                power_ratio = float(self.power_credit) / self.max_power_credit
                if power_ratio == 1 and f < 2:
                    pacman_color = pacman_normal_color
                    pacman_ec = None
                    
            elif self.dead and f > 0:
                pacman_color = "red"
                pacman_alpha = 0.4

            colors = [c for (_,c) in cells.values()]
            cmap = ListedColormap(colors)
            ax.pcolormesh(Z[::-1], cmap=cmap, edgecolors='k', alpha=0.8)
            
            # Add the power cookie(s)
            for (x,y) in power_cookies:
                power_cookie = mpatches.Circle((x,y), 0.22, color=power_cookie_color, ec="none")
                ax.add_artist(power_cookie)

            # Add the normal cookie(s)
            for (x,y) in cookies:
                cookie = mpatches.Circle((x,y), 0.17, color=cookie_color, ec="none")
                ax.add_artist(cookie)

            # Add the cookie that Pacman just ate in frame 0 and 1
            if self.n_steps > 0 and self.last_eaten_cookie['steps'] == self.n_steps and f < 2:
                i, j = self.last_eaten_cookie['pos']
                x = j + 0.5
                y = rows - i - 0.5
                if self.last_eaten_cookie['power'] == True:
                    power_cookie = mpatches.Circle((x,y), 0.22, color=power_cookie_color, ec="none")
                    ax.add_artist(power_cookie)
                else:
                    cookie = mpatches.Circle((x,y), 0.17, color=cookie_color, ec="none")
                    ax.add_artist(cookie)
                    
                
            # Add Pacman
            pacmans = []
            if f == 0:
                pacman_shift_x = 0.
                pacman_shift_y = 0.
                if self.pacman_heading != None:
                    pacman_shift_i = self.actions[self.pacman_heading][1][0]
                    pacman_shift_j = self.actions[self.pacman_heading][1][1]
                    pacman_shift_x = (pacman_shift_j) * 0.6
                    pacman_shift_y = (pacman_shift_i) * 0.6
                x = pacman_x - pacman_shift_x
                y = pacman_y + pacman_shift_y
                if power_ratio != 0:
                    pacman_alpha = power_ratio
                    normal_pacman = mpatches.Wedge((x, y), *pacman_open[0], **pacman_open[1],
                                                   color=pacman_normal_color,
                                                   ec=pacman_normal_ec,
                                                   alpha=(1.-power_ratio))
                    pacmans.append(normal_pacman)
                pacman = mpatches.Wedge((x, y), *pacman_open[0], **pacman_open[1],
                                        color=pacman_color, ec=pacman_ec, alpha=pacman_alpha)
                pacmans.append(pacman)
                
            elif f == 1:
                pacman_shift_x = 0.
                pacman_shift_y = 0.
                if self.pacman_heading != None:
                    pacman_shift_i = self.actions[self.pacman_heading][1][0]
                    pacman_shift_j = self.actions[self.pacman_heading][1][1]
                    pacman_shift_x = (pacman_shift_j) * 0.3
                    pacman_shift_y = (pacman_shift_i) * 0.3
                x = pacman_x - pacman_shift_x
                y = pacman_y + pacman_shift_y
                # Pacman with close mouth
                if power_ratio != 0:
                    pacman_alpha = power_ratio
                    normal_pacman = mpatches.Circle((x, y), 0.3,
                                                    color=pacman_normal_color,
                                                    ec=pacman_normal_ec,
                                                    alpha=(1.-power_ratio))
                    pacmans.append(normal_pacman)
                pacman = mpatches.Circle((x, y), 0.3,
                                         color=pacman_color, ec=pacman_ec, alpha=pacman_alpha)
                pacmans.append(pacman)
            else: # f == 2
                if power_ratio != 0:
                    pacman_alpha = power_ratio
                    normal_pacman = mpatches.Wedge((pacman_x, pacman_y), *pacman_open[0], **pacman_open[1],
                                                     color=pacman_normal_color, ec=pacman_normal_ec,
                                                     alpha=(1.-power_ratio))
                    pacmans.append(normal_pacman)
                pacman = mpatches.Wedge((pacman_x, pacman_y), *pacman_open[0], **pacman_open[1],
                                        color=pacman_color, ec=pacman_ec, alpha= pacman_alpha)
                pacmans.append(pacman)

            if self.dead == False or f < 2:
                for pacman in pacmans:
                    ax.add_artist(pacman)

            # Add the just eaten ghosts (if any) in the 2 first frames
            if eaten_ghosts != [] and f < 2:
                # Just display 1 ghost even is there is more than 1
                x, y = eaten_ghosts[0]
                color = edible_ghost_color
                w = mpatches.FancyBboxPatch((x-0.01, y+0.01), 0.2, -0.12, ec="none", color='white',
                                            boxstyle=mpatches.BoxStyle("Roundtooth", pad=0.12))
                ghost = mpatches.FancyBboxPatch((x,y), 0.2, 0.2, ec="black", color=color,
                                                boxstyle=mpatches.BoxStyle("Roundtooth", pad=0.3))
                ghost_eye_left = mpatches.Circle((x-0.05,y+0.15), 0.07, ec="none", color="white")
                ghost_eye_right = mpatches.Circle((x+0.2,y+0.15), 0.07, ec="none", color="white")
                ax.add_artist(ghost)
                ax.add_artist(ghost_eye_left)
                ax.add_artist(ghost_eye_right)
                ax.add_artist(w)
            
                 
            # Add the ghosts
            for (x,y, color, heading) in ghosts:
                    
                # Ghosts eyes color to be randomly selected from the list
                ghost_eye_color = ["white","white","white","white","white"]

                shift_x = 0
                shift_y = 0

                if f == 0:
                    ghost_eye_color = ["white","white","black","black","black"]
                    if heading != None:
                        shift_i = self.actions[heading][1][0]
                        shift_j = self.actions[heading][1][1]
                        shift_x = (shift_j) * 0.7
                        shift_y = (shift_i) * 0.7
                elif f == 1:
                    ghost_eye_color = ["white","white","black","black","black"]
                    if heading != None:
                        shift_i = self.actions[heading][1][0]
                        shift_j = self.actions[heading][1][1]
                        shift_x = (shift_j) * 0.3
                        shift_y = (shift_i) * 0.3

                x -= shift_x
                y += shift_y
                
                w = None

                if self.power_credit != 0:
                    if just_ate_power_cookie == False or f == 2:
                        color = edible_ghost_color
                        w = mpatches.FancyBboxPatch((x-0.01, y+0.01), 0.2, -0.12, ec="none", color='white',
                                                    boxstyle=mpatches.BoxStyle("Roundtooth", pad=0.12))
                    
                ghost = mpatches.FancyBboxPatch((x,y), 0.2, 0.2, ec="black", color=color,
                                                boxstyle=mpatches.BoxStyle("Roundtooth", pad=0.3))
                eyes_color = np.random.choice(ghost_eye_color)
                ghost_eye_left = mpatches.Circle((x-0.05,y+0.15), 0.07, ec="none", color=eyes_color)
                ghost_eye_right = mpatches.Circle((x+0.2,y+0.15), 0.07, ec="none", color=eyes_color)
                
                ax.add_artist(ghost)
                ax.add_artist(ghost_eye_left)
                ax.add_artist(ghost_eye_right)
                if w != None:
                    ax.add_artist(w)

            plt.pause(0.12)
            ax.set_axis_off()
            #display(fig);
            if f != 2:
                ax.cla();