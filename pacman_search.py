#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Look-ahead search guided by a trained agent.

When a ghost is close to Pacman, each possible move is evaluated by simulating
the rest of the game on copies of the environment:
    - the candidate move is played,
    - then the trained agent plays alone (greedy policy) for at most 'horizon' moves,
    - the simulation is scored with the rewards received plus, if the game is not
      over, the agent's estimate of the value of the state reached.
Two ways of choosing the move from the 'n_simulations' simulations of each move:
    - "safety" (default): the agent's move is kept, unless Pacman dies in clearly more
      simulations with it than with another move (by at least 'margin', e.g. 0.15 = 15% of
      the simulations): then the move with the fewest deaths is played (the best average
      score among them). The search only corrects the agent when it walks into a trap.
    - "value": the move with the best average score is played. With few simulations the
      averages are noisy and this changes many good moves of the agent by chance.
The ghosts' random choices differ between simulations (that is what is averaged), but
simulation k uses the same random numbers for every candidate move, so that the
moves are compared on the same possible futures.

The search runs on copies of the environment and does not consume the random
numbers of the real game: with or without search, the ghosts make the same
random draws, only Pacman's moves differ. The simulations never see the real
future: they use their own random numbers.

How the ghosts move in the simulations ('ghost_model'):
    - "exact": with the game's own code. The search then knows the exact behaviour
      of the ghosts (A* chase with their configured aggressiveness, random moves
      otherwise, fleeing when Pacman has power): a perfect model of the game.
    - "chase": worst case, without using the game's code: when Pacman has no power,
      every ghost moves along a shortest path towards Pacman at every move. When he
      has power, the ghosts move randomly (no assumption on how edible ghosts behave).
    - "estimated": without using the game's code: when Pacman has no power, a ghost
      moves along a shortest path towards Pacman with probability p_chase, and to a
      random neighbouring cell otherwise. When he has power, it moves away from him
      with probability p_flee, and randomly otherwise. p_chase and p_flee are estimated
      for each ghost from the moves it made since the beginning of the game (see
      GhostEstimator).
"""

import numpy as np
import torch
from collections import deque
from pacman_agents import A2C_PacmanAgent, DQN_PacmanAgent


def neighbour_cells(env, pos):
    '''Cells reachable from 'pos' in one move (including staying), without duplicates.'''
    i, j = pos
    return sorted(set(tuple(env.map_actions[i, j, idx]) for idx, _ in env.actions.values()))


def distance_table(env):
    '''
    Shortest-path distances (in moves) between all the cells of the board (breadth-first search).
    Returns a dict: dist[a][b].
    '''
    dist = {}
    for start in env.valid_pos:
        d = {start: 0}
        frontier = [start]
        while frontier:
            nxt = []
            for c in frontier:
                for n in neighbour_cells(env, c):
                    if n not in d:
                        d[n] = d[c] + 1
                        nxt.append(n)
            frontier = nxt
        dist[start] = d
    return dist


class GhostEstimator:
    '''
    Estimates, for each ghost, from its observed moves:
        - p_chase: probability that it moves towards Pacman when he has no power,
        - p_flee: probability that it moves away from Pacman when he has power.
    Model of a move: towards (resp. away from) Pacman with probability p, else to a random
    neighbouring cell (uniformly). A move "towards" (resp. "away from") Pacman decreases
    (resp. increases) the distance to Pacman's position before or after his move (the
    ghosts can move before or after Pacman).
    p is the maximum a posteriori estimate with a Beta(2, 2) prior (p = 0.5 without observation).
    '''
    def __init__(self, dist):
        self.dist = dist
        self.grid = np.linspace(0., 1., 101)
        self.reset()

    def reset(self):
        self.log_post = {}
        self.n_obs = {}

    def _towards(self, cell, ghost_pos, pacman_positions):
        return any(self.dist[cell][p] < self.dist[ghost_pos][p] for p in pacman_positions)

    def _away(self, cell, ghost_pos, pacman_positions):
        return any(self.dist[cell][p] > self.dist[ghost_pos][p] for p in pacman_positions)

    def observe(self, env, before):
        '''
            env: the real game after a move.
            before: (Pacman's position, power credit, {ghost id: position}) before the move.
        '''
        pacman_before, power_before, ghosts_before = before
        if env.dead:
            return
        if power_before == 0 and env.power_credit == 0:
            kind, rule = "chase", self._towards
        elif power_before > 1 and env.power_credit > 0:
            kind, rule = "flee", self._away
        else:
            return # the power starts or ends during this move
        pacman_positions = (pacman_before, env.current_pacman_pos)
        for ghost in env.ghosts:
            if ghost.id not in ghosts_before:
                continue
            g0 = ghosts_before[ghost.id]
            if g0 in pacman_positions:
                continue
            cells = neighbour_cells(env, g0)
            matching = [c for c in cells if rule(c, g0, pacman_positions)]
            q = len(matching) / len(cells)  # probability that a random move matches
            if q in (0., 1.):
                continue # not informative
            key = (kind, ghost.id)
            if key not in self.log_post:
                self.log_post[key] = np.log(self.grid * (1 - self.grid) + 1e-12) # Beta(2, 2) prior
                self.n_obs[key] = 0
            if tuple(ghost.pos) in matching:
                self.log_post[key] += np.log(self.grid + (1 - self.grid) * q)
            else:
                self.log_post[key] += np.log((1 - self.grid) * (1 - q) + 1e-12)
            self.n_obs[key] += 1

    def p(self, ghost_id, kind="chase"):
        key = (kind, ghost_id)
        if key not in self.log_post:
            return 0.5
        return float(self.grid[np.argmax(self.log_post[key])])


def snapshot(env):
    '''What GhostEstimator.observe() needs to know about the game before a move.'''
    return (env.current_pacman_pos, env.power_credit, {g.id: tuple(g.pos) for g in env.ghosts})


class LookaheadSearch:
    def __init__(self, agent, n_simulations=20, horizon=15, radius=5, mode="safety", margin=0.15,
                 ghost_model="chase"):
        '''
            agent: trained A2C_PacmanAgent or DQN_PacmanAgent (its first environment is the real game).
            n_simulations: number of simulations per candidate move.
            horizon: maximum number of moves simulated after the candidate move.
            radius: the search is used only when a ghost is at most this distance
                    (number of cells, horizontally + vertically) from Pacman.
            mode: "safety" or "value" (see the module description).
            margin: in "safety" mode, minimal difference of death rate to change the agent's move.
            ghost_model: "exact", "chase" or "estimated" (see the module description).
        '''
        if mode not in ("safety", "value"):
            raise ValueError(f"Unknown mode '{mode}'")
        if ghost_model not in ("exact", "chase", "estimated"):
            raise ValueError(f"Unknown ghost model '{ghost_model}'")
        self.ghost_model = ghost_model
        self.dist = distance_table(agent.pacman_envs[0]) if ghost_model != "exact" else None
        self.estimator = GhostEstimator(self.dist) if ghost_model == "estimated" else None
        self.mode = mode
        self.margin = margin
        self.agent = agent
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.radius = radius
        if isinstance(agent, A2C_PacmanAgent):
            self.gamma = agent.A2C.gamma
            self.reward_scale = agent.A2C.reward_scale
        elif isinstance(agent, DQN_PacmanAgent):
            self.gamma = agent.DQN.gamma
            self.reward_scale = agent.DQN.reward_scale
        else:
            raise ValueError("Unknown agent")
        self.n_searches = 0 # number of moves decided by the search
        self.n_changed = 0  # ... for which the search chose another move than the agent

    def new_game(self):
        '''To call at the beginning of each game.'''
        if self.estimator is not None:
            self.estimator.reset()

    def observe(self, env, before):
        '''To call after each move of the real game, with snapshot(env) taken before the move.'''
        if self.estimator is not None:
            self.estimator.observe(env, before)

    def _model_ghost(self, ghost, p_chase, p_flee):
        '''
        Replace the move of a simulated ghost by the generic model: when Pacman has
        no power, towards him with probability p_chase; when he has power, away from
        him with probability p_flee; random move otherwise.
        Uses only np.random (for the common random numbers of the simulations).
        '''
        dist = self.dist
        def move(env):
            pos = tuple(ghost.pos)
            cells = neighbour_cells(env, pos)
            pacman = env.current_pacman_pos
            if env.power_credit > 0:
                if np.random.random() < p_flee:
                    far = max(dist[c][pacman] for c in cells)
                    choices = [c for c in cells if dist[c][pacman] == far]
                else:
                    choices = cells
            elif np.random.random() < p_chase:
                near = min(dist[c][pacman] for c in cells)
                choices = [c for c in cells if dist[c][pacman] == near]
            else:
                choices = cells
            new_pos = choices[np.random.randint(len(choices))]
            ghost.heading = env.heading(np.asarray(pos), np.asarray(new_pos))
            ghost.pos = new_pos
        ghost.move = move # attribute of this copy of the ghost only

    def _evaluate(self, states):
        '''
        Greedy actions and state values (in game reward units) for a batch of states.
        '''
        states = torch.from_numpy(states)
        if isinstance(self.agent, A2C_PacmanAgent):
            with torch.no_grad():
                logits, values = self.agent.A2C.a2c_net(states.to(self.agent.A2C.device))
            actions = logits.argmax(dim=1)
        else:
            q = self.agent.DQN.qvalues(states)
            values, actions = q.max(dim=1)
        return actions.cpu().numpy(), values.cpu().numpy() / self.reward_scale

    def ghost_nearby(self, env):
        i, j = env.current_pacman_pos
        return any(abs(i - gi) + abs(j - gj) <= self.radius for (gi, gj) in (g.pos for g in env.ghosts))

    def candidate_moves(self, env):
        '''
        Index of the moves leading to different cells (moving into a wall is the same as staying).
        '''
        i, j = env.current_pacman_pos
        moves, targets = [], set()
        for a, (idx, _) in env.actions.items():
            target = tuple(env.map_actions[i, j, idx])
            if target not in targets:
                targets.add(target)
                moves.append(self.agent.actions.index(a))
        return moves

    def select_action(self, env, frames, agent_action):
        '''
            env: the real game.
            frames: the agent's current frame stack (deque of the last observations).
            agent_action: index of the move the agent would play without search.
        Returns the index of the move to play.
        '''
        if env.ghosts == [] or not self.ghost_nearby(env):
            return agent_action
        moves = self.candidate_moves(env)
        if len(moves) == 1:
            return moves[0]
        # The agent's move, as one of the candidates (same target cell)
        i, j = env.current_pacman_pos
        target = lambda m: tuple(env.map_actions[i, j, env.actions[self.agent.actions[m]][0]])
        agent_move = next(m for m in moves if target(m) == target(agent_action))

        real_rng_state = np.random.get_state()
        sim_seeds = np.random.randint(0, 2**31 - 1, size=self.n_simulations)

        # One simulation per (move, k)
        sims = []
        for move in moves:
            for k in range(self.n_simulations):
                sim_env = env.clone()
                if self.ghost_model != "exact":
                    for ghost in sim_env.ghosts:
                        if self.ghost_model == "chase":
                            self._model_ghost(ghost, p_chase=1., p_flee=0.)
                        else:
                            self._model_ghost(ghost, p_chase=self.estimator.p(ghost.id, "chase"),
                                              p_flee=self.estimator.p(ghost.id, "flee"))
                sims.append({"move": move, "env": sim_env, "frames": deque(frames, maxlen=frames.maxlen),
                             "rng": np.random.RandomState(sim_seeds[k]).get_state(),
                             "score": 0., "discount": 1.})

        def step(sim, action_idx):
            np.random.set_state(sim["rng"])
            reward = sim["env"].step(self.agent.actions[action_idx])
            sim["rng"] = np.random.get_state()
            sim["score"] += sim["discount"] * reward
            sim["discount"] *= self.gamma
            if not sim["env"].end:
                sim["frames"].append(self.agent.observation_of(sim["env"]))

        for sim in sims:
            step(sim, sim["move"])

        for t in range(self.horizon + 1):
            live = [sim for sim in sims if not sim["env"].end]
            if not live:
                break
            states = np.stack([np.concatenate(sim["frames"], axis=0) for sim in live])
            actions, values = self._evaluate(states)
            if t == self.horizon:
                # End of the simulation: value of the state reached
                for sim, v in zip(live, values):
                    sim["score"] += sim["discount"] * v
                break
            for sim, a in zip(live, actions):
                step(sim, int(a))

        np.random.set_state(real_rng_state)

        self.n_searches += 1
        scores = {move: np.mean([s["score"] for s in sims if s["move"] == move]) for move in moves}
        if self.mode == "value":
            best = max(moves, key=lambda m: scores[m])
            # Keep the agent's move when it is (one of) the best
            if scores[agent_move] >= scores[best] - 1e-9:
                return agent_action
        else:
            deaths = {move: np.mean([s["env"].dead for s in sims if s["move"] == move]) for move in moves}
            safest = min(deaths.values())
            if deaths[agent_move] - safest < self.margin:
                return agent_action
            best = max((m for m in moves if deaths[m] == safest), key=lambda m: scores[m])
        self.n_changed += 1
        return best
