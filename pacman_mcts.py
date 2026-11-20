'''
Monte Carlo tree search (MCTS) for Pacman, the ghosts being part of the environment.

Pacman is the only player: the ghosts do not choose their moves to beat him, they follow the
random rules of the game (each ghost chases on a move with the probability of its aggressiveness
and moves at random otherwise). The ghosts are therefore chance, as dice in backgammon, not an
opponent as in Hex: the value of a move is an average over the ghosts' random behaviours, not a
minimum.

The search is "open-loop": the tree holds only Pacman's move sequences. Each simulation starts
from a copy of the real game (env.clone()), with its own random numbers, and plays Pacman's
moves down the tree while the ghosts move by the game's own code. A node's statistics are thus
an average over many possible ghost behaviours. Pacman's moves are deterministic (only the
ghosts are random), so the same move sequence always leads Pacman to the same cells.

Each simulation:
    - descends the tree with PUCT (the agent's policy gives the priors, as in AlphaZero):
          score(a) = Q(a) / value_scale + c_puct P(a) sqrt(N + 1) / (1 + N(a)),
    - at a new node: the agent plays greedily for 'horizon' moves, then its value estimates
      the rest of the game (an empty horizon: the value at once, as in AlphaZero),
    - backs up the discounted return of each move of its path, the reward of the move
      included (a move onto a ghost ends the game: its -35 must count).
The most visited move is played.

Speed: the simulations run in waves of 'batch' simulations. A "virtual loss" makes the
simulations of a wave take different branches, and their network evaluations are made
in one batch. Each simulation keeps its own random state, so the waves do not change the
ghosts' behaviour in the simulations, and the search saves and restores the random state of
the real game: with or without search, the real ghosts make the same random draws (the games
of an evaluation are the same games).

Ghost models (ghost_model):
    exact:     the simulated ghosts move by the game's own code: the search knows their rules
               (each ghost's aggressiveness, the shortest-path chase), though not the random
               draws of the real game. A perfect model, for an estimate of what the game allows.
    estimated: the fair version: the simulated ghosts follow the generic model of
               pacman_search.py (towards Pacman along a shortest path with probability p_chase,
               away from him with probability p_flee when he has power, at random otherwise),
               with p_chase and p_flee estimated for each ghost from its moves since the start
               of the game (GhostEstimator: every game starts from no information).

Used by pacman_play.py (--mcts N) and pacman_expert.py (--teacher mcts).
'''
import math
from collections import deque
import numpy as np
import torch
from pacman_search import GhostEstimator, LookaheadSearch, distance_table


class Node:
    __slots__ = ("N", "W", "P", "children")

    def __init__(self, prior):
        self.N = 0           # number of visits
        self.W = 0.          # total return of the move that leads to this node
        self.P = prior       # prior probability of that move (agent's policy)
        self.children = None # {move index: Node} once expanded


class MCTS:
    def __init__(self, agent, n_simulations=100, horizon=10, radius=6, c_puct=1.5,
                 value_scale=35., batch=16, virtual_loss=10., ghost_model="exact"):
        '''
        agent: the A2C agent (policy for the priors and the rollouts, value at the leaves).
        n_simulations: simulations per search.
        horizon: moves played by the agent after a new node before using its value.
        radius: search only when a ghost is at most this distance (|di| + |dj|) from Pacman.
        c_puct: weight of the exploration term; value_scale: scale of the returns in Q
            (the penalty for being caught, 35, makes Q of order 1).
        batch: simulations per wave (network evaluations in one batch).
        virtual_loss: return subtracted from a move while a simulation of the wave goes
            through it, so that the other simulations of the wave try other moves.
        ghost_model: "exact" or "estimated" (see the module description).
        '''
        if ghost_model not in ("exact", "estimated"):
            raise ValueError(f"Unknown ghost model '{ghost_model}'")
        self.ghost_model = ghost_model
        self.dist = distance_table(agent.pacman_envs[0]) if ghost_model == "estimated" else None
        self.estimator = GhostEstimator(self.dist) if ghost_model == "estimated" else None
        self.agent = agent
        self.n_simulations, self.horizon, self.radius = n_simulations, horizon, radius
        self.c_puct, self.value_scale = c_puct, value_scale
        self.batch, self.virtual_loss = batch, virtual_loss
        self.gamma = agent.A2C.gamma
        self.reward_scale = agent.A2C.reward_scale
        self.n_searches = 0
        self.n_changed = 0
        self.last = None   # statistics of the last search: {move index: (visits, Q, prior)}

    # Interface of pacman_play.play_game (as LookaheadSearch)
    def new_game(self):
        if self.estimator is not None:
            self.estimator.reset()

    def observe(self, env, before):
        if self.estimator is not None:
            self.estimator.observe(env, before)

    def _clone(self, env):
        '''A copy of the real game for a simulation, its ghosts following the chosen model.'''
        sim = env.clone()
        if self.estimator is not None:
            # (eaten ghosts and ghosts in the house too: they come back during the simulation)
            for ghost in sim.ghosts + sim.dead_ghosts + sim.house_ghosts:
                LookaheadSearch._model_ghost(self, ghost, p_chase=self.estimator.p(ghost.id, "chase"),
                                             p_flee=self.estimator.p(ghost.id, "flee"))
        return sim

    def _evaluate(self, frame_lists):
        '''Policy (probabilities) and value (in rewards) of a batch of frame stacks.'''
        states = np.stack([np.concatenate(list(f), axis=0) for f in frame_lists])
        with torch.no_grad():
            logits, values = self.agent.A2C.a2c_net(torch.from_numpy(states).to(self.agent.A2C.device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs, values.view(-1).cpu().numpy() / self.reward_scale

    def _moves(self, env):
        '''Indices of the moves that lead to different cells.'''
        i, j = env.current_pacman_pos
        moves, targets = [], set()
        for idx, a in enumerate(self.agent.actions):
            t = tuple(env.map_actions[i, j, env.actions[a][0]])
            if t not in targets:
                targets.add(t)
                moves.append(idx)
        return moves

    def _target(self, env, m):
        i, j = env.current_pacman_pos
        return tuple(env.map_actions[i, j, env.actions[self.agent.actions[m]][0]])

    def ghost_near(self, env):
        i, j = env.current_pacman_pos
        return any(abs(i - gi) + abs(j - gj) <= self.radius for gi, gj in (g.pos for g in env.ghosts))

    def _expand(self, node, env, probs):
        moves = self._moves(env)
        total = sum(probs[m] for m in moves) or 1.
        node.children = {m: Node(probs[m] / total) for m in moves}

    def search(self, env, frames):
        '''
        Run the simulations from the real game 'env' (frames: the agent's frame stack).
        Returns {move index: (visits, Q, prior)} for the moves of the root.
        '''
        real_state = np.random.get_state()
        seeds = np.random.randint(0, 2**31 - 1, size=self.n_simulations)
        root = Node(1.)
        probs, _ = self._evaluate([frames])
        self._expand(root, env, probs[0])
        done = 0
        while done < self.n_simulations:
            wave = []
            for k in range(done, min(done + self.batch, self.n_simulations)):
                np.random.seed(seeds[k])
                sim = self._clone(env)
                fr = deque(frames, maxlen=frames.maxlen)
                node, path, rewards = root, [root], []
                while not sim.end and node.children is not None:
                    sq = math.sqrt(node.N + 1)
                    best, best_score = None, -float("inf")
                    for m, ch in node.children.items():
                        q = ch.W / ch.N / self.value_scale if ch.N else 0.
                        s = q + self.c_puct * ch.P * sq / (1 + ch.N)
                        if s > best_score:
                            best, best_score = m, s
                    child = node.children[best]
                    child.N += 1                      # virtual loss, removed at the backup
                    child.W -= self.virtual_loss
                    node.N += 1
                    rewards.append(sim.step(self.agent.actions[best]))
                    if not sim.end:
                        fr.append(self.agent.observation_of(sim))
                    node = child
                    path.append(node)
                wave.append({"sim": sim, "frames": fr, "path": path, "rewards": rewards,
                             "rng": np.random.get_state(), "ret": 0., "disc": 1.})
            done += len(wave)

            # Rollouts of the wave, the network evaluations in one batch per move
            active = [w for w in wave if not w["sim"].end]
            for h in range(self.horizon + 1):
                if not active:
                    break
                probs, values = self._evaluate([w["frames"] for w in active])
                still = []
                for w, p, v in zip(active, probs, values):
                    if h == 0 and w["path"][-1].children is None:
                        self._expand(w["path"][-1], w["sim"], p)   # priors of the new node
                    if h == self.horizon:
                        w["ret"] += w["disc"] * v                  # value of the rest of the game
                        continue
                    np.random.set_state(w["rng"])
                    r = w["sim"].step(self.agent.actions[int(np.argmax(p))])
                    w["rng"] = np.random.get_state()
                    w["ret"] += w["disc"] * r
                    w["disc"] *= self.gamma
                    if not w["sim"].end:
                        w["frames"].append(self.agent.observation_of(w["sim"]))
                        still.append(w)
                active = still

            # Backup (and removal of the virtual losses)
            for w in wave:
                g = w["ret"]
                path, rewards = w["path"], w["rewards"]
                for d in range(len(path) - 1, 0, -1):
                    g = rewards[d - 1] + self.gamma * g
                    path[d].W += g + self.virtual_loss
        np.random.set_state(real_state)
        self.n_searches += 1
        self.last = {m: (ch.N, ch.W / max(1, ch.N), ch.P) for m, ch in root.children.items()}
        return self.last

    def select_action(self, env, frames, agent_action):
        '''The move to play: the agent's move far from the ghosts, the most visited move near them.'''
        if not env.ghosts or not self.ghost_near(env):
            return agent_action
        if len(self._moves(env)) == 1:
            return agent_action
        stats = self.search(env, frames)
        best = max(stats, key=lambda m: (stats[m][0], stats[m][1]))
        if self._target(env, best) == self._target(env, agent_action):
            return agent_action
        self.n_changed += 1
        return best
