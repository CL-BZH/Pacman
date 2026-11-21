'''
Paired evaluation of the MCTS on game v2: the agent alone, the MCTS with the exact ghost model
and the MCTS with the estimated (fair) ghost model, on the same games.

The games are played in batches of 10 (as pacman_play.py), in parallel on the CPU cores. The
batches of the two searches are interleaved, so that the results stay paired if the run is
stopped early. Each batch is appended to OUT (one JSON line per game) as soon as it ends, and
a batch already in OUT is not replayed: the run can be restarted where it stopped.

Usage (from the repository's root):
    python3 Tools/mcts_eval.py CHECKPOINT [--games 100] [--seed 4001] [--sims 100]
                               [--horizon 10] [--batch 1] [--workers 4] [--frames 2]
                               [--configs agent,exact,estimated] [--out mcts_eval.jsonl]
Then:
    python3 Tools/mcts_eval.py CHECKPOINT --summary --out mcts_eval.jsonl
'''
import sys, os, json, time, argparse, multiprocessing
from math import comb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

_players = {}


def player(config, ckpt, sims, horizon, batch):
    '''The player of a configuration ("agent", "exact" or "estimated"), made once per process.'''
    if config not in _players:
        import torch, pacman_play as pp
        from pacman_mcts import MCTS
        torch.set_num_threads(1)
        sys.argv = ["pacman_play.py", "-g", "v2", "-a", "a2c", "-f", os.environ.get("MCTS_EVAL_FRAMES", "2")]
        args = pp.parse_args()
        env, agent, stockastic, _ = pp.make_player(args, ckpt, quiet=True)
        search = None if config == "agent" else MCTS(agent, n_simulations=sims, horizon=horizon,
                                                     batch=batch, ghost_model=config)
        _players[config] = (pp, args, (env, agent, stockastic, search))
    return _players[config]


def play(task):
    config, first, n, base, ckpt, sims, horizon, batch = task
    pp, args, pl = player(config, ckpt, sims, horizon, batch)
    pl[1].weights = np.ones_like(pl[1].weights)   # as pacman_play: weights reset per batch
    search = pl[3]
    s0, c0 = (search.n_searches, search.n_changed) if search else (0, 0)
    t = time.time()
    records, *_ = pp.play_games(pl, range(first, min(first + 10, n)), base, args, progress=False)
    out = []
    for r in records:
        out.append({"config": config, "batch": first, "seed": int(r["seed"]),
                    "won": bool(not r["lost"] and not r["aborted"]),
                    "seconds": (time.time() - t) / len(records)})
    if search:
        out[0]["searches"], out[0]["changed"] = search.n_searches - s0, search.n_changed - c0
    return out


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sign_test(x, y):
    '''Two-sided sign test on x against y disagreements.'''
    n = x + y
    return 1. if n == 0 else min(1., 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n)


def summary(path):
    recs = load(path)
    res = {}
    for r in recs:
        res.setdefault(r["config"], {})[r["seed"]] = r["won"]
    configs = [c for c in ("agent", "exact", "estimated") if c in res]
    seeds = sorted(set.intersection(*(set(res[c]) for c in configs))) if configs else []
    print(f"{len(seeds)} games played by all of {', '.join(configs)}")
    for c in configs:
        w = sum(res[c][s] for s in seeds)
        n = len(seeds)
        se = 100 * np.sqrt(w / n * (1 - w / n) / n) if n else 0.
        print(f"  {c:9s} won {w}/{n} = {100 * w / max(1, n):.1f}% (+/- {se:.1f})")
    for a, b in (("exact", "agent"), ("estimated", "agent"), ("exact", "estimated")):
        if a in res and b in res:
            x = sum(res[a][s] and not res[b][s] for s in seeds)
            y = sum(res[b][s] and not res[a][s] for s in seeds)
            print(f"  {a} wins {x} games that {b} loses, {b} wins {y} that {a} loses: "
                  f"sign test p = {sign_test(x, y):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=4001)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--batch", type=int, default=1, help="simulations per wave (1: one at a time)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--frames", type=int, default=2, help="frames stacked by the agent's network")
    p.add_argument("--configs", default="agent,exact,estimated")
    p.add_argument("--out", default="mcts_eval.jsonl")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        summary(a.out)
        sys.exit()
    os.environ["MCTS_EVAL_FRAMES"] = str(a.frames)   # (read by the worker processes)
    done = {(r["config"], r["batch"]) for r in load(a.out)}
    configs = a.configs.split(",")
    # The agent alone first (fast), then the two searches interleaved batch by batch
    tasks = [(c, k, a.games, a.seed, a.checkpoint, a.sims, a.horizon, a.batch)
             for c in ([x for x in configs if x == "agent"]) for k in range(0, a.games, 10)]
    tasks += [(c, k, a.games, a.seed, a.checkpoint, a.sims, a.horizon, a.batch)
              for k in range(0, a.games, 10) for c in configs if c != "agent"]
    tasks = [t for t in tasks if (t[0], t[1]) not in done]
    print(f"{len(tasks)} batches of 10 games to play on {a.workers} workers", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for recs in pool.imap_unordered(play, tasks):
            with open(a.out, "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            r = recs[0]
            print(f"[{(time.time() - t0) / 60:.0f} min] {r['config']} games {r['batch'] + 1}-"
                  f"{r['batch'] + len(recs)}: won {sum(x['won'] for x in recs)}/{len(recs)}, "
                  f"{r['seconds']:.0f} s per game", flush=True)
    summary(a.out)
