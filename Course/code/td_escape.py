"""
Tabular Q-learning and SARSA on the reduced Pacman game of chapter 3
(course, chapter 4). The agents only sample transitions; the exact
solution of dp_escape.py is used to evaluate them.
"""
import numpy as np
import dp_escape as d

rng = np.random.default_rng(0)
LIVE = np.array([k for k, (p, g) in enumerate(d.S) if p != g and p != d.GOAL])


def sample(k, a):
    """Sample (reward, next state or -1) from the transition of (k, a)."""
    out = d.T[k][a]
    j = rng.choice(len(out), p=[pr for pr, _, _ in out])
    return out[j][1], out[j][2]


def eps_greedy(Q, k, eps):
    if rng.random() < eps:
        return int(rng.integers(d.nA))
    q = Q[k]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def run(method, episodes, eps=0.1, alpha=0.2, checkpoints=()):
    Q = np.zeros((d.nS, d.nA))
    results = []
    for e in range(1, episodes + 1):
        k = int(rng.choice(LIVE))
        a = eps_greedy(Q, k, eps)
        for _ in range(500):
            r, k2 = sample(k, a)
            if k2 < 0:
                Q[k, a] += alpha * (r - Q[k, a])
                break
            a2 = eps_greedy(Q, k2, eps)
            if method == "q-learning":
                target = r + d.GAMMA * Q[k2].max()
            else:  # sarsa
                target = r + d.GAMMA * Q[k2, a2]
            Q[k, a] += alpha * (target - Q[k, a])
            k, a = k2, a2
        if e in checkpoints:
            pi = Q.argmax(axis=1)
            ps = d.success_probability(pi)
            results.append((e, ps[LIVE].mean()))
    return Q, results


if __name__ == "__main__":
    v_star, _ = d.value_iteration()
    cps = (1_000, 5_000, 20_000, 50_000, 100_000, 200_000)
    for method in ("q-learning", "sarsa"):
        Q, res = run(method, cps[-1], checkpoints=cps)
        err = np.abs(Q.max(axis=1) - v_star)[LIVE]
        print(method, " ".join(f"{e}:{p:.3f}" for e, p in res),
              f"| mean |max_a Q - v*| = {err.mean():.3f}, max = {err.max():.2f}",
              flush=True)
