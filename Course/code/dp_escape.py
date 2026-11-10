"""
Dynamic programming on a reduced Pacman game (course, chapter 4).

Maze pacman_map3 of game v1, Pacman and ONE ghost (aggressiveness 0.74),
no cookies. Pacman must reach a power cookie placed in a corner; reaching it
gives +10 and ends the episode, being eaten gives -35 and ends it, and each
move costs -0.15. Order of a step (as for the ghosts 1, 2, ... of the game):
Pacman moves, then the ghost; Pacman is eaten if both are on the same cell
after either move. The ghost chases with probability 0.74 (to a next cell of
a shortest path, uniformly if there are several), and otherwise moves
uniformly to a cell reachable in one move (including its own cell).

The state is (Pacman's cell, ghost's cell): 77 x 77 = 5929 states.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pacman_maps import pacman_map3 as M

GAMMA, LIVING, GOAL_R, DEAD_R, AGGR = 0.99, -0.15, 10.0, -35.0, 0.74
MOVES = {"E": (0, 1), "N": (-1, 0), "W": (0, -1), "S": (1, 0), "I": (0, 0)}
ACTIONS = list(MOVES)
cells = sorted(M.rooms)
idx = {c: k for k, c in enumerate(cells)}
n = len(cells)
GOAL = (9, 13)                       # bottom right corner of the maze

def step(c, d):
    t = (c[0] + d[0], c[1] + d[1])
    return t if t in M.rooms else c

nbrs = [sorted({step(c, d) for d in MOVES.values()}) for c in cells]
# maze distances (BFS)
dist = np.full((n, n), np.inf)
for k, c in enumerate(cells):
    dist[k, k] = 0; front = [c]
    while front:
        new = []
        for x in front:
            for y in nbrs[idx[x]]:
                if dist[k, idx[y]] == np.inf:
                    dist[k, idx[y]] = dist[k, idx[x]] + 1; new.append(y)
        front = new

def ghost_dist(g, p):
    """Distribution of the ghost's next cell (ghost at g, Pacman at p)."""
    out = {}
    nb = nbrs[idx[g]]
    best = min(dist[idx[x], idx[p]] for x in nb)
    chase = [x for x in nb if dist[idx[x], idx[p]] == best]
    for x in chase:
        out[x] = out.get(x, 0) + AGGR / len(chase)
    for x in nb:
        out[x] = out.get(x, 0) + (1 - AGGR) / len(nb)
    return out

S = [(p, g) for p in cells for g in cells]
sid = {s: k for k, s in enumerate(S)}
nS, nA = len(S), len(ACTIONS)
# Transitions: for each (s, a) a list of (prob, reward, next state or -1 if terminal)
T = [[None] * nA for _ in range(nS)]
for k, (p, g) in enumerate(S):
    for a, act in enumerate(ACTIONS):
        if p == g or p == GOAL:       # terminal states (absorbing, value 0)
            T[k][a] = [(1.0, 0.0, -1)]; continue
        p2 = step(p, MOVES[act])
        if p2 == g:
            T[k][a] = [(1.0, LIVING + DEAD_R, -1)]; continue
        if p2 == GOAL:
            T[k][a] = [(1.0, LIVING + GOAL_R, -1)]; continue
        out = []
        for g2, pr in ghost_dist(g, p2).items():
            if g2 == p2:
                out.append((pr, LIVING + DEAD_R, -1))
            else:
                out.append((pr, LIVING, sid[(p2, g2)]))
        T[k][a] = out

def backup(v, k, a):
    return sum(pr * (r + (GAMMA * v[s2] if s2 >= 0 else 0.0)) for pr, r, s2 in T[k][a])

def value_iteration(tol=1e-8):
    v = np.zeros(nS); it = 0
    while True:
        it += 1
        new = np.array([max(backup(v, k, a) for a in range(nA)) for k in range(nS)])
        delta = np.max(np.abs(new - v)); v = new
        if delta < tol:
            return v, it

def greedy(v):
    return np.array([max(range(nA), key=lambda a: backup(v, k, a)) for k in range(nS)])

def evaluate(pi, tol=1e-10):
    v = np.zeros(nS)
    while True:
        new = np.array([backup(v, k, pi[k]) for k in range(nS)])
        if np.max(np.abs(new - v)) < tol:
            return new
        v = new

def policy_iteration():
    pi = np.full(nS, ACTIONS.index("I")); it = 0
    while True:
        it += 1
        v = evaluate(pi)
        new = greedy(v)
        if np.array_equal(new, pi):
            return pi, v, it
        pi = new

def success_probability(pi, tol=1e-12):
    """Probability of reaching the goal (gamma = 1, reward 1 at the goal)."""
    v = np.zeros(nS)
    while True:
        new = np.zeros(nS)
        for k in range(nS):
            p, g = S[k]
            if p == g or p == GOAL:
                continue
            for pr, r, s2 in T[k][pi[k]]:
                new[k] += pr * ((1.0 if r > 0 else 0.0) if s2 < 0 else v[s2])
        if np.max(np.abs(new - v)) < tol:
            return new
        v = new

def expected_moves(pi, tol=1e-9):
    """Expected number of moves until the end of the episode under pi."""
    v = np.zeros(nS)
    while True:
        new = np.zeros(nS)
        for k in range(nS):
            p, g = S[k]
            if p == g or p == GOAL:
                continue
            new[k] = sum(pr * (1 + (v[s2] if s2 >= 0 else 0)) for pr, r, s2 in T[k][pi[k]])
        if np.max(np.abs(new - v)) < tol:
            return new
        v = new


if __name__ == "__main__":
    print("Solving the reduced game by value and policy iteration (about 15 seconds)...",
          flush=True)
    v, it_vi = value_iteration()
    pi_vi = greedy(v)
    pi_pi, v_pi, it_pi = policy_iteration()
    print(f"{nS} states; value iteration: {it_vi} sweeps; "
          f"policy iteration: {it_pi} iterations")
    print(f"max |v_VI - v_PI| = {np.max(np.abs(v - v_pi)):.1e}; "
          f"the policies differ in {np.sum(pi_vi != pi_pi)} states")
    # Naive policy: shortest path to the goal, ignoring the ghost
    naive = np.array([min(range(nA), key=lambda a: (
        dist[idx[step(p, MOVES[ACTIONS[a]])], idx[GOAL]], a)) for (p, g) in S])
    ps_opt, ps_naive = success_probability(pi_vi), success_probability(naive)
    live = [k for k, (p, g) in enumerate(S) if p != g and p != GOAL]
    print(f"success probability over the non-terminal states: optimal "
          f"min {ps_opt[live].min():.4f} mean {ps_opt[live].mean():.4f}; "
          f"shortest path mean {ps_naive[live].mean():.4f}")
    k = sid[((5, 7), (7, 12))]
    em_opt = expected_moves(pi_vi)
    print(f"start Pacman (5, 7), ghost (7, 12): v* = {v[k]:.2f}; success optimal "
          f"{ps_opt[k]:.3f} in {em_opt[k]:.1f} moves on average, "
          f"shortest path {ps_naive[k]:.3f}")
