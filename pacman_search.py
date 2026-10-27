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
random draws, only Pacman's moves differ.
"""

import numpy as np
import torch
from collections import deque
from pacman_agents import A2C_PacmanAgent, DQN_PacmanAgent


class LookaheadSearch:
    def __init__(self, agent, n_simulations=20, horizon=15, radius=5, mode="safety", margin=0.15):
        '''
            agent: trained A2C_PacmanAgent or DQN_PacmanAgent (its first environment is the real game).
            n_simulations: number of simulations per candidate move.
            horizon: maximum number of moves simulated after the candidate move.
            radius: the search is used only when a ghost is at most this distance
                    (number of cells, horizontally + vertically) from Pacman.
            mode: "safety" or "value" (see the module description).
            margin: in "safety" mode, minimal difference of death rate to change the agent's move.
        '''
        if mode not in ("safety", "value"):
            raise ValueError(f"Unknown mode '{mode}'")
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
                sims.append({"move": move, "env": env.clone(), "frames": deque(frames, maxlen=frames.maxlen),
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
