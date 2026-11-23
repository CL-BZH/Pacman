'''
End-game heat maps of several agents on the same games (game v2), and how similar they are.

Record (per agent): plays N games greedily, as pacman_play.py -g v2 -n N --seed SEED (same seeds,
batches of 10), and saves for each game whether it was lost, the cells of the last 10 cookies
(when the game reaches 10 cookies left) and Pacman's cell at the end:
    python3 Tools/endgame_maps.py record CHECKPOINT OUT.json [--frames 2] [--games 1000]
                                         [--seed 1] [--workers 4]
Compare (several agents):
    python3 Tools/endgame_maps.py compare FIGURE.png name1=OUT1.json name2=OUT2.json ...
The figure shows, for each agent, how often each cell is among the last 10 cookies (% of the games
that reach 10 cookies left) and where Pacman is caught (% of the lost games). Printed:
    - the Pearson correlation between the agents' maps, over the open cells of the maze,
    - the overlap of their lost games (Jaccard index), with what independent losses would give,
    - the games lost (won) by all the agents, observed and expected if the losses were independent.
'''
import sys, os, json, argparse, itertools, multiprocessing
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

_player = None


def _init(ckpt, frames):
    global _player
    import torch, pacman_play as pp
    torch.set_num_threads(1)
    sys.argv = ["pacman_play.py", "-g", "v2", "-a", "a2c", "-f", str(frames)]
    args = pp.parse_args()
    _player = (pp, args, pp.make_player(args, ckpt, quiet=True))


def _left(env):
    return len(env.cookies) + len(env.power_cookies)


def _play(task):
    games, base_seed = task
    pp, args, (env, agent, stockastic, _) = _player
    agent.weights = np.ones_like(agent.weights)   # as pacman_play: weights reset per batch
    out = []
    for i in games:
        pp.set_seed(base_seed + i)
        agent.reset_episode()
        state = agent.current_state[0]
        last10, moves = None, 0
        while not env.end and moves <= args.max_moves:
            _, action = agent.select_action(state, stockastic=stockastic and env.n_ghosts_in_game == 0)
            env.step(action)
            state = agent.observe(0)
            moves += 1
            if last10 is None and _left(env) <= 10:
                last10 = [[int(x) for x in c] for c in list(env.cookies) + list(env.power_cookies)]
        out.append({"game": i + 1, "lost": bool(env.dead), "aborted": not env.end, "last10": last10,
                    "end_pos": [int(x) for x in env.current_pacman_pos], "left": _left(env)})
    return out


def record(a):
    tasks = [(range(k, min(k + 10, a.games)), a.seed) for k in range(0, a.games, 10)]
    recs = []
    with multiprocessing.get_context("spawn").Pool(a.workers, initializer=_init,
                                                   initargs=(a.checkpoint, a.frames)) as pool:
        for r in pool.imap_unordered(_play, tasks):
            recs += r
    recs.sort(key=lambda r: r["game"])
    json.dump(recs, open(a.out, "w"))
    print(f"won {100 * np.mean([not r['lost'] and not r['aborted'] for r in recs]):.1f}%")


def compare(a):
    from pacman_maps import pacman_map4 as M
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    agents = [(s.split("=")[0], json.load(open(s.split("=")[1]))) for s in a.agents]
    R, C = M.rows, M.columns
    walls, house = (M.map == 1), (M.map == 2)
    open_ = ~(walls | house)

    def maps(recs):
        reached = [r for r in recs if r["last10"] is not None]
        lost = [r for r in recs if r["lost"]]
        last, caught = np.zeros((R, C)), np.zeros((R, C))
        for r in reached:
            for i, j in r["last10"]:
                last[i, j] += 1
        for r in lost:
            caught[tuple(r["end_pos"])] += 1
        return 100 * last / max(1, len(reached)), 100 * caught / max(1, len(lost)), len(reached), len(lost)

    m = {name: maps(recs) for name, recs in agents}
    names = [n for n, _ in agents]
    fig, ax = plt.subplots(2, len(agents), figsize=(3.3 * len(agents), 7.4), squeeze=False)
    vmax = [max(m[n][k].max() for n in names) for k in (0, 1)]
    for col, (name, recs) in enumerate(agents):
        won = 100 * np.mean([not r["lost"] and not r["aborted"] for r in recs])
        for row, (cmap, title) in enumerate([
                ("YlOrRd", f"{name} ({won:.1f}% won)\nlast 10 cookies ({m[name][2]} games)"),
                ("Blues", f"where caught ({m[name][3]} games)")]):
            ax[row, col].imshow(np.ma.masked_where(~open_, m[name][row]), cmap=cmap, vmin=0, vmax=vmax[row])
            ax[row, col].imshow(np.ma.masked_where(~walls, walls), cmap=ListedColormap(["#333333"]))
            ax[row, col].imshow(np.ma.masked_where(~house, house), cmap=ListedColormap(["#bbbbbb"]))
            ax[row, col].set_title(title, fontsize=9)
            ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
    plt.tight_layout()
    plt.savefig(a.figure, dpi=110)

    for k, label in [(0, "last-10 maps"), (1, "catch maps")]:
        print(f"Pearson correlation of the {label} (over the {open_.sum()} open cells):")
        print("        " + " ".join(f"{b:>7s}" for b in names))
        for x in names:
            print(f"{x:7s} " + " ".join(f"{np.corrcoef(m[x][k][open_], m[y][k][open_])[0, 1]:7.2f}"
                                         for y in names))
    n = len(agents[0][1])
    lost = {name: {r["game"] for r in recs if r["lost"]} for name, recs in agents}
    won = {name: {r["game"] for r in recs if not r["lost"] and not r["aborted"]} for name, recs in agents}
    print("Lost games in common: observed, and expected if the losses were independent")
    for x, y in itertools.combinations(names, 2):
        both = len(lost[x] & lost[y])
        exp = len(lost[x]) * len(lost[y]) / n
        print(f"  {x} / {y}: {both} (expected {exp:.0f}), Jaccard {both / len(lost[x] | lost[y]):.2f} "
              f"(expected {exp / (len(lost[x]) + len(lost[y]) - exp):.2f})")
    print(f"Lost by all: {len(set.intersection(*lost.values()))} "
          f"(expected {n * np.prod([len(v) / n for v in lost.values()]):.0f}); "
          f"won by all: {len(set.intersection(*won.values()))} "
          f"(expected {n * np.prod([len(v) / n for v in won.values()]):.0f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("checkpoint"); r.add_argument("out")
    r.add_argument("--frames", type=int, default=2); r.add_argument("--games", type=int, default=1000)
    r.add_argument("--seed", type=int, default=1); r.add_argument("--workers", type=int, default=os.cpu_count())
    c = sub.add_parser("compare")
    c.add_argument("figure"); c.add_argument("agents", nargs="+")
    a = p.parse_args()
    record(a) if a.command == "record" else compare(a)
