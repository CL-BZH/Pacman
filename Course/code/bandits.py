"""
Bernoulli bandit testbed (course, chapter 2): epsilon-greedy, UCB1 and
Thompson sampling on 10 arms with random means, averaged over runs.
"""
import numpy as np

K, T, RUNS = 10, 100_000, 100
CHECKPOINTS = (100, 1_000, 10_000, 100_000)
rng = np.random.default_rng(0)


def run(policy, mu):
    n = np.zeros(K); s = np.zeros(K); regret = 0.0; out = []
    for t in range(1, T + 1):
        if policy == "greedy":
            a = int(np.argmax(np.where(n > 0, s / np.maximum(n, 1), np.inf)))
        elif policy.startswith("eps"):
            eps = float(policy[3:])
            if rng.random() < eps or n.min() == 0:
                a = int(rng.integers(K)) if n.min() > 0 else int(np.argmin(n))
            else:
                a = int(np.argmax(s / n))
        elif policy == "ucb1":
            if t <= K:
                a = t - 1
            else:
                a = int(np.argmax(s / n + np.sqrt(2 * np.log(t) / n)))
        elif policy == "thompson":
            a = int(np.argmax(rng.beta(1 + s, 1 + n - s)))
        r = float(rng.random() < mu[a])
        n[a] += 1; s[a] += r
        regret += mu.max() - mu[a]
        if t in CHECKPOINTS:
            out.append(regret)
    return out


if __name__ == "__main__":
    policies = ["greedy", "eps0.1", "eps0.01", "ucb1", "thompson"]
    res = {p: [] for p in policies}
    for _ in range(RUNS):
        mu = rng.random(K)
        for p in policies:
            res[p].append(run(p, mu))
    print("regret after", CHECKPOINTS, "steps (mean over", RUNS, "runs)")
    for p in policies:
        r = np.array(res[p])
        print(f"{p:9s}", "  ".join(f"{m:8.1f}" for m in r.mean(axis=0)))
