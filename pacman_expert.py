#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert iteration: train the A2C agent with a search as teacher.

Two teachers (--teacher):
    lookahead: the short look-ahead search of pacman_search.py (game v1); its target is the
               move it would play.
    mcts:      the Monte Carlo tree search of pacman_mcts.py (game v2, whose ghosts the short
               search cannot beat); its target is the distribution of its visits over the moves
               (as in AlphaZero), which also tells how good the other moves are.

Each iteration:
    1. Collection: the current agent plays games alone (in parallel processes).
       At each move where a ghost is close, the teacher gives its target. As the games are played by the agent, the search
       labels exactly the situations the agent gets into, including its mistakes
       (DAgger, Ross et al., 2011; expert iteration, Anthony et al., 2017).
    2. Training: the policy learns the teacher's target where a ghost is close
       (with a larger weight where the search disagrees with the agent), and its own
       move elsewhere (so that the rest of its play does not change). The value head
       learns the discounted returns observed in the games.
The games of each collection are played without search: their results measure the
progress of the agent alone.

Examples:
    python pacman_expert.py --iterations 10 --games 1000 --workers 4
    python pacman_expert.py -g v2 -c Trainings/A2C/PPO_v2/pacman_a2c_final.pth --teacher mcts \
        --iterations 5 --games 200 --workers 4
    python pacman_play.py -c Tmp/expert/pacman_expert_final.pth -n 1000
"""

import argparse
import contextlib
import io
import multiprocessing
import os
import pathlib
import random
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pacman_training import GAMES
from pacman_env import PacmanEnvironment
from pacman_agents import A2C_PacmanAgent
from pacman_search import LookaheadSearch, snapshot
from pacman_mcts import MCTS

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(REPO_DIR, "Trainings", "A2C", "PPO_1", "pacman_a2c_final.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Expert iteration: train the A2C agent with the "
                                                 "look-ahead search as teacher.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--checkpoint", default=DEFAULT_CHECKPOINT, help="A2C agent to start from")
    parser.add_argument("-g", "--game", choices=list(GAMES), default="v1",
                        help="game version (see pacman_training.py); must be the one the agent was trained on")
    parser.add_argument("--iterations", type=int, default=10, help="number of iterations")
    parser.add_argument("--games", type=int, default=1000, help="games played per iteration")
    parser.add_argument("-w", "--workers", type=int, default=4, help="processes playing games in parallel")
    parser.add_argument("--teacher", choices=["lookahead", "mcts"], default="lookahead",
                        help="the search used as teacher (see above)")
    parser.add_argument("--mcts-sims", type=int, default=100, help="mcts: simulations per move")
    parser.add_argument("--mcts-horizon", type=int, default=10,
                        help="mcts: moves played by the agent after a new node before its value")
    parser.add_argument("--mcts-radius", type=int, default=6, help="mcts: used when a ghost is at most this distance")
    parser.add_argument("--sims", type=int, default=10, help="lookahead: simulations per candidate move")
    parser.add_argument("--horizon", type=int, default=10, help="lookahead: moves simulated after each candidate move")
    parser.add_argument("--radius", type=int, default=5, help="lookahead: used when a ghost is at most this distance")
    parser.add_argument("--ghost-model", choices=["exact", "chase", "estimated"], default="chase",
                        help="lookahead: how the ghosts move in the simulations")
    parser.add_argument("--changed-weight", type=float, default=5.,
                        help="weight of the moves where the search disagrees with the agent")
    parser.add_argument("--other-fraction", type=float, default=0.25,
                        help="fraction of the moves without search (no ghost close) used for training")
    parser.add_argument("--keep", type=int, default=2, help="train on the data of the last KEEP iterations")
    parser.add_argument("--epochs", type=int, default=2, help="passes over the data per iteration")
    parser.add_argument("--batch", type=int, default=512, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--value-coef", type=float, default=0.5, help="weight of the value loss")
    parser.add_argument("--eval-games", type=int, default=1000,
                        help="games played without search after the last iteration (0 = no final evaluation)")
    parser.add_argument("--max-moves", type=int, default=800, help="games longer than this are aborted")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("-o", "--out", default="./Tmp/expert", help="output folder")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collection (in worker processes)

# The states are stored as uint8: the feature planes are binary except Pacman's
# power (power_credit / max_power_credit), so multiplying by max_power_credit gives integers.
def power_scale(args):
    return GAMES[args.game].max_power_credit

_agent = None
_search = None
_loaded = None
_args = None


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def _init_worker(args, device):
    global _agent, _search, _args
    torch.set_num_threads(1)
    _args = args
    env = PacmanEnvironment(GAMES[args.game])
    _agent = _quiet(A2C_PacmanAgent, [env], device=device)
    if args.teacher == "mcts":
        _search = MCTS(_agent, n_simulations=args.mcts_sims, horizon=args.mcts_horizon, radius=args.mcts_radius)
    else:
        _search = LookaheadSearch(_agent, n_simulations=args.sims, horizon=args.horizon, radius=args.radius,
                                  mode="safety", ghost_model=args.ghost_model)


def _load(checkpoint):
    global _loaded
    if _loaded != checkpoint:
        _agent.load(checkpoint)
        _loaded = checkpoint


def teacher_target(agent, search, env, action):
    '''
    The teacher's target at a move where a ghost is close: a probability distribution over the
    agent's actions (one-hot for the look-ahead search, the visit distribution for the MCTS),
    and the move the teacher would play.
    '''
    n_actions = len(agent.actions)
    target = np.zeros(n_actions, dtype=np.float32)
    if isinstance(search, MCTS):
        if len(search._moves(env)) == 1:
            target[action] = 1.
            return target, action
        stats = search.search(env, agent.frame_stacks[0])
        for m, (visits, _, _) in stats.items():
            target[m] = visits
        target /= target.sum()
        best = max(stats, key=lambda m: (stats[m][0], stats[m][1]))
        if search._target(env, best) == search._target(env, action):
            best = action   # same cell as the agent's move: not a disagreement
        return target, best
    move = search.select_action(env, agent.frame_stacks[0], action)
    target[move] = 1.
    return target, move


def play_labelled_game(agent, search, seed, args, label=True):
    '''
    The agent plays one game alone. If 'label', the teacher gives its target at each
    move where a ghost is close. Returns the game's records and result.
    '''
    env = agent.pacman_envs[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent.weights = np.ones_like(agent.weights) # start position depends only on the seed
    agent.reset_episode()
    search.new_game()
    states, targets, labels, changed, searched, rewards = [], [], [], [], [], []
    n_actions = len(agent.actions)
    near = search.ghost_near if isinstance(search, MCTS) else search.ghost_nearby
    state = agent.current_state[0]
    while not env.end and env.n_steps < args.max_moves:
        action, _ = agent.select_action(state, stockastic=(env.ghosts == []))
        policy, move, is_searched = None, action, False
        if label and env.ghosts and near(env):
            policy, move = teacher_target(agent, search, env, action)
            is_searched = True
        if policy is None:
            policy = np.zeros(n_actions, dtype=np.float32)
            policy[action] = 1.
        states.append(np.rint(state[0].numpy() * power_scale(args)).astype(np.uint8))
        targets.append(policy)
        labels.append(move)
        changed.append(move != action)
        searched.append(is_searched)
        before = snapshot(env)
        rewards.append(env.step(agent.actions[action])) # the agent's own move is played
        search.observe(env, before)
        state = agent.observe(0)

    # Discounted returns (in the network's units), bootstrapped if the game was aborted
    gamma, scale = agent.A2C.gamma, agent.A2C.reward_scale
    g = 0. if env.end else float(agent.A2C.values(state.numpy())[0])
    returns = np.zeros(len(rewards), dtype=np.float32)
    for t in reversed(range(len(rewards))):
        g = rewards[t] * scale + gamma * g
        returns[t] = g

    # Keep all the moves with search, and a fraction of the others
    keep_rng = np.random.default_rng(seed)
    keep = [s or keep_rng.random() < args.other_fraction for s in searched]
    idx = [t for t, k in enumerate(keep) if k]
    data = {"states": np.array([states[t] for t in idx], dtype=np.uint8).reshape(len(idx), *agent.input_shape),
            "labels": np.array([labels[t] for t in idx], dtype=np.int64),
            "targets": np.array([targets[t] for t in idx], dtype=np.float32).reshape(len(idx), n_actions),
            "changed": np.array([changed[t] for t in idx], dtype=bool),
            "returns": returns[idx]}
    result = {"won": env.end and not env.dead, "lost": env.dead, "aborted": not env.end,
              "searched": int(np.sum(searched)), "changed": int(np.sum(changed)), "moves": env.n_steps}
    return data, result


def _worker_collect(task):
    checkpoint, seeds, label = task
    _load(checkpoint)
    out = [play_labelled_game(_agent, _search, s, _args, label) for s in seeds]
    return out


def collect(pool, checkpoint, seeds, label, progress=""):
    batch = 10
    tasks = [(checkpoint, seeds[k:k + batch], label) for k in range(0, len(seeds), batch)]
    data, results = [], []
    next_report = len(seeds) // 10
    for out in pool.imap_unordered(_worker_collect, tasks):
        for d, r in out:
            data.append(d)
            results.append(r)
        if progress and len(results) >= next_report and len(results) < len(seeds):
            print(f"{progress}{len(results)}/{len(seeds)} games", flush=True)
            next_report += max(1, len(seeds) // 10)
    return data, results


# ---------------------------------------------------------------------------
# Training (in the main process)

def train(net, optimizer, datasets, args, device):
    states = np.concatenate([d["states"] for d in datasets])
    labels = np.concatenate([d["labels"] for d in datasets])
    targets = np.concatenate([d["targets"] for d in datasets])
    changed = np.concatenate([d["changed"] for d in datasets])
    returns = np.concatenate([d["returns"] for d in datasets])
    weights = np.where(changed, args.changed_weight, 1.).astype(np.float32)
    n = len(labels)
    net.train()
    stats = {"policy_loss": 0., "value_loss": 0., "accuracy_changed": 0.}
    n_batches = 0
    for _ in range(args.epochs):
        perm = np.random.permutation(n)
        for k in range(0, n, args.batch):
            idx = perm[k:k + args.batch]
            x = torch.from_numpy(states[idx].astype(np.float32) / power_scale(args)).to(device)
            y = torch.from_numpy(targets[idx]).to(device)
            w = torch.from_numpy(weights[idx]).to(device)
            ret = torch.from_numpy(returns[idx]).to(device)
            logits, values = net(x)
            # Cross-entropy with the teacher's distribution (one-hot: the usual cross-entropy)
            policy_loss = (-(y * F.log_softmax(logits, dim=1)).sum(dim=1) * w).sum() / w.sum()
            value_loss = 0.5 * (values - ret).pow(2).mean()
            loss = policy_loss + args.value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            n_batches += 1
    # How often the network now plays the search's move where the search disagreed with the agent
    net.eval()
    if changed.any():
        with torch.no_grad():
            x = torch.from_numpy(states[changed].astype(np.float32) / power_scale(args)).to(device)
            pred = net(x)[0].argmax(dim=1).cpu().numpy()
        stats["accuracy_changed"] = float(np.mean(pred == labels[changed]))
    stats["policy_loss"] /= max(1, n_batches)
    stats["value_loss"] /= max(1, n_batches)
    stats["samples"] = n
    return stats


def summary(results):
    n = len(results)
    return {"games": n,
            "won_%": 100 * sum(r["won"] for r in results) / n,
            "lost_%": 100 * sum(r["lost"] for r in results) / n,
            "aborted_%": 100 * sum(r["aborted"] for r in results) / n,
            "searched_per_game": sum(r["searched"] for r in results) / n,
            "changed_per_game": sum(r["changed"] for r in results) / n}


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = PacmanEnvironment(GAMES[args.game])
    agent = _quiet(A2C_PacmanAgent, [env], device=str(device))
    agent.load(args.checkpoint)
    net = agent.A2C.a2c_net
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    teacher = (f"MCTS, {args.mcts_sims} simulations, horizon {args.mcts_horizon}, radius {args.mcts_radius}"
               if args.teacher == "mcts" else
               f"look-ahead search, {args.sims} simulations, horizon {args.horizon}, {args.ghost_model} ghosts")
    print(f"Expert iteration from {args.checkpoint} on {device}: {args.iterations} iterations of "
          f"{args.games} games, {args.workers} workers; teacher: {teacher}", flush=True)

    checkpoint = str(out / "pacman_expert_0.pth")
    agent.save(checkpoint)
    log, datasets = [], []
    start = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args, str(device))) as pool:
        for it in range(1, args.iterations + 1):
            seeds = [args.seed * 10_000_000 + it * 100_000 + g for g in range(args.games)]
            t0 = time.time()
            data, results = collect(pool, checkpoint, seeds, label=True, progress=f"Iteration {it}: ")
            s = summary(results)
            datasets.append({k: np.concatenate([d[k] for d in data]) for k in data[0]})
            datasets = datasets[-args.keep:]
            t1 = time.time()
            stats = train(net, optimizer, datasets, args, device)
            checkpoint = str(out / f"pacman_expert_{it}.pth")
            agent.save(checkpoint)
            row = {"iteration": it, **s, **stats,
                   "collect_s": round(t1 - t0), "train_s": round(time.time() - t1)}
            log.append(row)
            pd.DataFrame(log).to_csv(out / "expert_log.csv", index=False)
            print(f"{timedelta(seconds=int(time.time() - start))} | iteration {it:2d} | agent alone "
                  f"(network of iteration {it - 1}): won {s['won_%']:5.1f}% lost {s['lost_%']:4.1f}% | "
                  f"search: {s['searched_per_game']:.0f} moves/game, changed {s['changed_per_game']:.2f}/game | "
                  f"policy loss {stats['policy_loss']:.3f}, value loss {stats['value_loss']:.3f}, "
                  f"search moves learnt {100 * stats['accuracy_changed']:.0f}% | "
                  f"collect {row['collect_s']} s, train {row['train_s']} s", flush=True)

        agent.save(str(out / "pacman_expert_final.pth"))
        if args.eval_games > 0:
            seeds = [10**9 + g for g in range(args.eval_games)]
            _, results = collect(pool, checkpoint, seeds, label=False, progress="Final evaluation: ")
            s = summary(results)
            print(f"Final agent alone ({args.eval_games} games, no search): won {s['won_%']:.1f}%, "
                  f"lost {s['lost_%']:.1f}%, aborted {s['aborted_%']:.1f}%", flush=True)
            log.append({"iteration": "final", **s})
            pd.DataFrame(log).to_csv(out / "expert_log.csv", index=False)
    print(f"Done in {timedelta(seconds=int(time.time() - start))}. Agent: {out / 'pacman_expert_final.pth'}")


if __name__ == "__main__":
    main()
