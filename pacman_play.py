#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:24:25 2024

@author: chris
"""

"""
Let a trained agent play Pacman.

Examples:
    python pacman_play.py                       # display one game of the default A2C agent
    python pacman_play.py -n 1000               # statistics over 1000 games
    python pacman_play.py -a dqn -c my_dqn.pth  # DQN agent with another checkpoint
    python pacman_play.py -n 200 --show-losses  # replay (with display) up to 5 lost games out of 200
    python pacman_play.py -n 200 --search 20    # with look-ahead search when a ghost is close
    python pacman_play.py -n 1000 --search 20 -w 8  # same, 8 games in parallel
    python pacman_play.py --help                # all the options
"""

from pacman_training import pacman_conf  # same game configuration as for the training
from pacman_env import PacmanEnvironment
from pacman_agents import DQN_PacmanAgent
from pacman_agents import A2C_PacmanAgent
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
import argparse
import contextlib
import io
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pacman_search import LookaheadSearch, snapshot


# Default trained agents (relative to the repository folder)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINTS = {
    "a2c": os.path.join("Trainings", "A2C", "PPO_1", "pacman_a2c_final.pth"),
    "dqn": os.path.join("Trainings", "DQN", "pacman_dqn_final.pth"),
}


# Causes of the lost games (see loss_cause())
POWER_ENDED_MOVES = 3 # "power-ended": death at most this number of moves after the last move with power
BAD_START_MOVES = 5   # "bad-start": death in the first moves of the game
LOSS_CAUSES = {
    "off-by-one": "Pacman moved onto a ghost on the move where his power ran out",
    "power-ended": f"caught at most {POWER_ENDED_MOVES} moves after the power ran out (other cases)",
    "bad-start": f"caught in the first {BAD_START_MOVES} moves, without having had any power",
    "other": "all the other lost games",
}


def loss_cause(power_before, died_on_pacman_move):
    '''
    Cause of a lost game.
        power_before: Pacman's power credit before each move of the game (the last one is the fatal move).
        died_on_pacman_move: True if Pacman died by moving onto a ghost
                             (False if a ghost moved onto Pacman).
    The power credit is decreased at the beginning of Pacman's move, before the collision
    test: with 1 move of power left, moving onto a ghost kills Pacman ("off-by-one").
    '''
    n_moves = len(power_before)
    if power_before[-1] == 1 and died_on_pacman_move:
        return "off-by-one"
    moves_with_power = [t for t, p in enumerate(power_before) if p > 0]
    if moves_with_power and (n_moves - 1) - moves_with_power[-1] <= POWER_ENDED_MOVES:
        return "power-ended"
    if n_moves <= BAD_START_MOVES and not moves_with_power:
        return "bad-start"
    return "other"


def play_game(env, agent, stockastic=False, display=False, wait=15, speed=0.12, max_moves=800, search=None):
    '''
    Play one game.
        stockastic: if True, the action is sampled from the policy once all ghosts are eaten
                    (the greedy action is always used while ghosts are present).
        display: show the game in a window. The game is computed first, then shown at a
                 constant speed (moves that use the search take longer to compute).
        wait: seconds to wait before the game starts (display only).
        speed: pause between two moves in seconds (display only).
        max_moves: the game is aborted after this number of moves.
        search: optional LookaheadSearch (see pacman_search.py) used when a ghost is close.
    Returns (lost, aborted, number of ghosts left, number of power cookies eaten,
             cause of the loss (see loss_cause(), None if the game was not lost)).
    '''

    agent.reset_episode()
    if search is not None:
        search.new_game()
        
    # Copies of the game after each move, to show it once it is over
    frames = [env.clone()] if display else None
    if display and search is not None:
        print("Computing the game...", flush=True)
        
    rewards = []
    aborted = False

    # To find the cause of a loss: Pacman's power before each move
    power_before = []
    
    state = agent.current_state[0]
    while env.end == False:
        if env.ghosts != []:
            action_idx, action = agent.select_action(state, stockastic=False)
            if search is not None:
                action = agent.actions[search.select_action(env, agent.frame_stacks[0], action_idx)]
        else:
            _, action = agent.select_action(state, stockastic=stockastic)

        power_before.append(env.power_credit)
        before = snapshot(env) if search is not None else None
        reward = env.step(action)
        if search is not None:
            search.observe(env, before)
        state = agent.observe(0)
        agent.episode_rewards.append(reward)
        rewards.append(reward)
        
        if display:
            frames.append(env.clone())
                
        if len(rewards) == 600:
            print(f"\nNon finished. Number of ghost: {len(env.ghosts)}.")
        if len(rewards) > max_moves:
            print(f"\nAbort.Number of ghost: {len(env.ghosts)}.") #"\nRewards:\n{rewards}")
            aborted = True
            break
        
    if display:
        animate_game(frames, wait, speed)
    cause = loss_cause(power_before, env.died_on_pacman_move) if env.dead else None
    return env.dead == True, aborted, len(env.ghosts), env.n_eaten_power_cookies, cause


def animate_game(frames, wait, speed):
    '''
    Show a game from the copies of the environment taken after each move.
        wait: seconds to wait before the first move.
        speed: pause between two moves in seconds.
    '''
    env = frames[0]
    rows = env.state[0].shape[0]
    columns = env.state[0].shape[1]
    m = max(rows, columns)
    width = 5 * float(columns)/m
    height = 5 * float(rows)/m
    plt.ion()
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(width, height)
    ax.set_axis_off()
    env.colormap(width, height, fig, ax, speed)
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    plt.pause(max(wait, 0.01))
    for k, env in enumerate(frames[1:]):
        ax.cla()
        env.colormap(width, height, fig, ax, speed)
    plt.ioff()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Let a trained agent play Pacman. With one game the game is displayed; "
                    "with several games, statistics are printed at the end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--agent", choices=["a2c", "dqn"], default="a2c",
                        help="type of agent")
    parser.add_argument("-c", "--checkpoint", default=None,
                        help="network parameters (.pth); default: "
                             + ", ".join(f"{k}: {v}" for k, v in DEFAULT_CHECKPOINTS.items()))
    parser.add_argument("-n", "--games", type=int, default=1,
                        help="number of games to play")
    parser.add_argument("-f", "--frames", type=int, default=2,
                        help="number of stacked frames (must match the training)")
    parser.add_argument("--greedy", action="store_true",
                        help="always play the best action (by default the A2C agent samples its "
                             "action from the policy once all ghosts are eaten)")
    parser.add_argument("--display", action="store_true",
                        help="display the games even when several games are played")
    parser.add_argument("--wait", type=float, default=3.,
                        help="seconds to wait before a displayed game starts")
    parser.add_argument("--speed", type=float, default=0.12,
                        help="pause between two moves of a displayed game (seconds)")
    parser.add_argument("--max-moves", type=int, default=800,
                        help="games longer than this are aborted")
    parser.add_argument("--search", type=int, default=0, metavar="N",
                        help="look-ahead search with N simulations per move when a ghost is close "
                             "(see pacman_search.py); 0 = no search")
    parser.add_argument("--horizon", type=int, default=15,
                        help="with --search, number of moves simulated after each candidate move")
    parser.add_argument("--search-mode", choices=["safety", "value"], default="safety",
                        help="with --search: 'safety' changes the agent's move only when it clearly leads "
                             "to more deaths; 'value' plays the move with the best simulated score")
    parser.add_argument("--ghost-model", choices=["exact", "chase", "estimated"], default="chase",
                        help="with --search, how the ghosts move in the simulations: 'exact' = the game's "
                             "code, 'chase' = always towards Pacman (worst case), 'estimated' = towards "
                             "Pacman with a probability estimated from the ghosts' observed moves")
    parser.add_argument("--radius", type=int, default=5,
                        help="with --search, search only when a ghost is at most this distance from Pacman")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="number of games played in parallel (CPU processes); useful with --search")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (to replay the same games); game i uses seed + i")
    parser.add_argument("--show-losses", type=int, nargs="?", const=5, default=0, metavar="N",
                        help="after playing the games without display, replay up to N of the "
                             "lost games with display (N = 5 if not given)")
    parser.add_argument("--cause", choices=list(LOSS_CAUSES), default=None,
                        help="with --show-losses, replay only the games lost for this cause")
    return parser.parse_args()


def print_loss_causes(n_games, cause_count):
    n_lost = sum(cause_count.values())
    if n_lost == 0:
        return
    print(f"Causes of the {n_lost} lost games:")
    for cause, description in LOSS_CAUSES.items():
        v = cause_count[cause]
        print(f"\t{cause:<12} {v:5d} games: {100*v/n_games:5.2f}% of the games, "
              f"{100*v/n_lost:5.1f}% of the lost games  ({description})")


def print_statistics(n_games, lost, abort_count, ghost_count, eaten_power_cookies_count):
    if lost:
        winnings = (1. - np.sum(lost)/len(lost))*100.0
        print(f"Pacman won {winnings:.2f}% of the games")
    else:
        print("All the games were aborted")
    total_ghosts = 0.
    total_eaten_cookies = 0.
    print("Number of game aborted:") 
    for k, v in dict(sorted(abort_count.items())).items():
        print(f"\twith {k} ghost: {v}")
    print("Percentage of games with")
    for k, v in dict(sorted(ghost_count.items())).items():
        print(f"\t{k} ghost: {(v *100) / n_games}")
        total_ghosts += k * v
    print(f"Average number of ghost per game: {total_ghosts/float(n_games)}")
    print("Percentage of power cookies eaten per game:")
    for k, v in dict(sorted(eaten_power_cookies_count.items())).items():
        print(f"\t{k} power cookies: {(v *100) / float(n_games)}")
        total_eaten_cookies += k * v
    print(f"Average number of power cookies eaten per game: {total_eaten_cookies/float(n_games)}")


def make_player(args, checkpoint, quiet=False):
    '''
    Environment, agent (loaded from 'checkpoint'), policy mode and optional search.
    '''
    if quiet:
        # Do not print the agent's description in every worker process
        with contextlib.redirect_stdout(io.StringIO()):
            return make_player(args, checkpoint)
    # Same game configuration as for the training
    env = PacmanEnvironment(pacman_conf)
    if args.agent == "a2c":
        agent = A2C_PacmanAgent([env], n_frames=args.frames)
        stockastic = not args.greedy
    else:
        agent = DQN_PacmanAgent([env], n_frames=args.frames)
        agent.epsilon = 0.
        stockastic = False
    agent.load(checkpoint)
    search = None
    if args.search > 0:
        search = LookaheadSearch(agent, n_simulations=args.search, horizon=args.horizon, radius=args.radius,
                                 mode=args.search_mode, ghost_model=args.ghost_model)
    return env, agent, stockastic, search


def play_games(player, games, base_seed, args, display=False, progress=True):
    '''
    Play the games whose numbers are in 'games' (game i uses the seed base_seed + i).
    Returns one record (dict) per game, and the search counters.
    '''
    env, agent, stockastic, search = player
    records = []
    for i in games:
        if progress:
            print(f'\r{i+1}', end='\r', flush=True)
        # The agent's start-position weights change at every game: keep them to replay the game
        weights = agent.weights.copy()
        set_seed(base_seed + i)
        has_lost_game, aborted, n_ghost, n_eaten_power_cookies, cause = play_game(
            env, agent, stockastic, display, wait=args.wait, speed=args.speed, max_moves=args.max_moves,
            search=search)
        records.append({"game": i + 1, "seed": base_seed + i, "lost": has_lost_game, "aborted": aborted,
                        "ghosts": n_ghost, "power_cookies": n_eaten_power_cookies, "cause": cause,
                        "moves": env.n_steps, "weights": weights if has_lost_game else None})
    counters = (search.n_searches, search.n_changed) if search is not None else (0, 0)
    return records, counters


# Multiprocessing (--workers): each worker process loads its own agent once
_worker_player = None
_worker_args = None

def _init_worker(args, checkpoint):
    global _worker_player, _worker_args
    torch.set_num_threads(1)
    _worker_args = args
    _worker_player = make_player(args, checkpoint, quiet=True)

def _worker_play(task):
    games, base_seed = task
    # Batches are given to the workers in any order: reset the start-position weights
    # so that the games of a batch do not depend on the batches played before
    agent = _worker_player[1]
    agent.weights = np.ones_like(agent.weights)
    return play_games(_worker_player, games, base_seed, _worker_args, progress=False)


def main():
    args = parse_args()
    # Each game gets its own seed so that any game can be replayed identically
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(10**6)

    checkpoint = args.checkpoint or os.path.join(REPO_DIR, DEFAULT_CHECKPOINTS[args.agent])
    if not os.path.isfile(checkpoint):
        sys.exit(f"Checkpoint not found: {checkpoint}\n"
                 f"Use -c to give the checkpoint to load (a training saves its checkpoints "
                 f"in ./Tmp, e.g. -c Tmp/pacman_{args.agent}_final.pth).")

    player = make_player(args, checkpoint)
    env, agent, stockastic, search = player
    print(f"Loaded {checkpoint}")
    if search is not None:
        print(f"Look-ahead search ({args.search_mode}, {args.ghost_model} ghosts): {args.search} "
              f"simulations per move, horizon {args.horizon}, radius {args.radius}")

    display = (args.display or args.games == 1) and not args.show_losses
    workers = 1 if display else max(1, min(args.workers, args.games))

    if workers == 1:
        records, (n_searches, n_changed) = play_games(player, range(args.games), base_seed, args, display)
    else:
        # Games are distributed in small batches; results do not depend on the order of completion
        batch = 10
        tasks = [(range(k, min(k + batch, args.games)), base_seed) for k in range(0, args.games, batch)]
        records, n_searches, n_changed = [], 0, 0
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(args, checkpoint)) as pool:
            for recs, (s, c) in pool.imap_unordered(_worker_play, tasks):
                records += recs
                n_searches += s
                n_changed += c
                print(f'\r{len(records)}/{args.games}', end='\r', flush=True)
        records.sort(key=lambda r: r["game"])

    lost_games = [] # (game number, seed, start-position weights, moves, cause) of the lost games
    cause_count = defaultdict(int)
    lost = []
    abort_count = defaultdict(int)
    ghost_count = defaultdict(int)
    eaten_power_cookies_count = defaultdict(int)
    for r in records:
        if r["lost"]:
            cause_count[r["cause"]] += 1
            if args.cause is None or r["cause"] == args.cause:
                lost_games.append((r["game"], r["seed"], r["weights"], r["moves"], r["cause"]))
        if r["aborted"] == False:
            lost.append(r["lost"])
        ghost_count[r["ghosts"]] += 1.
        abort_count[r["ghosts"]] += int(r["aborted"])
        eaten_power_cookies_count[r["power_cookies"]] += 1.

    print("",end='\n')

    if args.games > 1:
        print_statistics(args.games, lost, abort_count, ghost_count, eaten_power_cookies_count)
        print_loss_causes(args.games, cause_count)
    else:
        r = records[0]
        result = "aborted" if r["aborted"] else (f"lost ({r['cause']})" if r["lost"] else "won")
        print(f"Game {result}. Ghosts left: {r['ghosts']}. Power cookies eaten: {r['power_cookies']}.")
    if n_searches:
        print(f"Search used for {n_searches} moves ({n_searches/args.games:.1f} per game); "
              f"it changed the agent's move {n_changed} times ({100*n_changed/n_searches:.1f}%)")
    workers_str = f" --workers {workers}" if workers > 1 else ""
    print(f"Seed: {base_seed} (use --seed {base_seed}{workers_str} to play the same games again)")

    if args.show_losses:
        replay_lost_games(env, agent, stockastic, lost_games[:args.show_losses], len(lost_games), args, search)
    elif args.cause:
        print("--cause is only used with --show-losses")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def replay_lost_games(env, agent, stockastic, lost_games, n_lost, args, search=None):
    '''
    Replay lost games with display. A game is replayed identically by restoring
    its random seed and the agent's start-position weights.
    '''
    if not lost_games:
        print("No game lost: nothing to replay.")
        return
    selection = f" with cause '{args.cause}'" if args.cause else ""
    print(f"{n_lost} game(s) lost{selection}. Replaying {len(lost_games)} of them "
          "(close the window at the end of a game to see the next one).")
    for game, seed, weights, n_steps, cause in lost_games:
        print(f"Game {game} (seed {seed}): lost after {n_steps} moves ({cause}: {LOSS_CAUSES[cause]})")
        agent.weights = weights.copy()
        set_seed(seed)
        has_lost_game, _, _, _, _ = play_game(env, agent, stockastic, display=True, wait=args.wait,
                                              speed=args.speed, max_moves=args.max_moves, search=search)
        if not has_lost_game or env.n_steps != n_steps:
            print("\tWarning: the replayed game differs from the original one.")


if __name__ == "__main__":
    main()
