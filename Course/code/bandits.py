"""
Bandit testbed (course, chapter 2): epsilon-greedy, UCB1 and Thompson
sampling on 10 arms with random means, averaged over runs.

    python bandits.py             # Bernoulli arms, the table of the course: 100 runs of
                                  # 100,000 rounds (50 million steps in Python: about 5-10 min)
    python bandits.py --quick     # 20 runs of 10,000 rounds: a few seconds
    python bandits.py --runs 50 --steps 100000
    python bandits.py --gaussian  # Gaussian arms (standard deviation 1): UCB for Gaussian
                                  # rewards, Gaussian Thompson sampling, and Thompson sampling
                                  # with a wrong model (standard deviation 0.1 instead of 1)
"""
import argparse
import time
import numpy as np

K, T, RUNS = 10, 100_000, 100
CHECKPOINTS = (100, 1_000, 10_000, 100_000)
rng = np.random.default_rng(0)


SIGMA = 1.0  # standard deviation of the rewards of the Gaussian arms


def run(policy, mu, gaussian=False):
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
        elif policy == "ucb-gauss":
            # UCB for Gaussian rewards of standard deviation SIGMA: the bonus scales with SIGMA
            if t <= K:
                a = t - 1
            else:
                a = int(np.argmax(s / n + SIGMA * np.sqrt(2 * np.log(t) / n)))
        elif policy.startswith("ts-gauss"):
            # Gaussian Thompson sampling, flat prior: theta_a ~ N(mean_a, sigma^2 / n_a).
            # "ts-gauss0.1" uses a wrong model of the rewards (sigma = 0.1 instead of 1)
            sigma_model = float(policy[8:]) if len(policy) > 8 else SIGMA
            if t <= K:
                a = t - 1
            else:
                a = int(np.argmax(rng.normal(s / n, sigma_model / np.sqrt(n))))
        if gaussian:
            r = mu[a] + SIGMA * rng.standard_normal()
        else:
            r = float(rng.random() < mu[a])
        n[a] += 1; s[a] += r
        regret += mu.max() - mu[a]
        if t in CHECKPOINTS:
            out.append(regret)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bandit testbed of chapter 2 of the course.")
    parser.add_argument("--runs", type=int, default=RUNS, help=f"number of runs (default {RUNS})")
    parser.add_argument("--steps", type=int, default=T, help=f"rounds per run (default {T})")
    parser.add_argument("--quick", action="store_true", help="20 runs of 10,000 rounds")
    parser.add_argument("--gaussian", action="store_true",
                        help="Gaussian arms (standard deviation 1) instead of Bernoulli arms")
    args = parser.parse_args()
    if args.quick:
        args.runs, args.steps = 20, 10_000
    RUNS, T = args.runs, args.steps
    CHECKPOINTS = tuple(c for c in CHECKPOINTS if c <= T)

    if args.gaussian:
        policies = ["greedy", "eps0.1", "eps0.01", "ucb-gauss", "ts-gauss", "ts-gauss0.1"]
    else:
        policies = ["greedy", "eps0.1", "eps0.01", "ucb1", "thompson"]
    res = {p: [] for p in policies}
    start = time.time()
    for i in range(RUNS):
        mu = rng.random(K)
        for p in policies:
            res[p].append(run(p, mu, args.gaussian))
        elapsed = time.time() - start
        left = elapsed / (i + 1) * (RUNS - i - 1)
        print(f"\rrun {i + 1}/{RUNS}  ({elapsed:.0f} s, about {left:.0f} s left)",
              end="", flush=True)
    print()
    print(("Gaussian" if args.gaussian else "Bernoulli"), "arms: regret after", CHECKPOINTS,
          "steps (mean over", RUNS, "runs)")
    for p in policies:
        r = np.array(res[p])
        print(f"{p:11s}", "  ".join(f"{m:8.1f}" for m in r.mean(axis=0)))
