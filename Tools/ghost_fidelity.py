'''
How well does the estimated ghost model predict the real ghosts' moves, compared with the true rules?

The agent plays games alone. At each move, before the real step, the transition is simulated K times
with the true game code (env.clone()) and K times with the estimated model (MCTS._clone with
ghost_model="estimated"), giving for each ghost two distributions of its next cell. The real step
then gives the observed next cell. Reported per ghost id, overall and for ghosts near Pacman:
    - mean log-likelihood of the observed cell under each model (Laplace smoothing),
    - mean total-variation distance between the two distributions.
Usage (from the repository's root):
    python3 Tools/ghost_fidelity.py CHECKPOINT N_GAMES BASE_SEED K WORKERS OUT.json
e.g. python3 Tools/ghost_fidelity.py Trainings/A2C/PPO_v2/pacman_a2c_100M.pth 100 4001 100 4 fidelity.json
'''
import sys, os, json, math, multiprocessing
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
from collections import Counter, defaultdict

_ctx = None


def init(ckpt):
    global _ctx
    import torch, pacman_play as pp
    from pacman_mcts import MCTS
    from pacman_search import snapshot, neighbour_cells
    torch.set_num_threads(1)
    sys.argv = ["pacman_play.py", "-g", "v2", "-a", "a2c"]
    args = pp.parse_args()
    env, agent, _, _ = pp.make_player(args, ckpt, quiet=True)
    est = MCTS(agent, ghost_model="estimated")
    _ctx = (env, agent, est, snapshot, neighbour_cells)


def next_cells(make, env, action, K, seeds):
    '''For each ghost id on the board before and after: Counter of its cell after the move.'''
    out = defaultdict(Counter)
    ids = {g.id for g in env.ghosts}
    for k in range(K):
        np.random.seed(seeds[k])
        sim = make(env)
        sim.step(action)
        if sim.dead:
            continue
        for g in sim.ghosts:
            if g.id in ids:
                out[g.id][tuple(int(c) for c in g.pos)] += 1
    return out


def play(task):
    game, base, K = task
    env, agent, est, snapshot, neighbour_cells = _ctx
    np.random.seed(base + game)

    agent.reset_episode()
    est.new_game()
    state = agent.current_state[0]
    rec = []
    n = 0
    while not env.end and n < 1000:
        n += 1
        _, action = agent.select_action(state, stockastic=False)
        if env.ghosts and env.power_credit == 0:
            real = np.random.get_state()
            seeds = np.random.randint(0, 2**31 - 1, size=K)
            true = next_cells(lambda e: e.clone(), env, action, K, seeds)
            gen = next_cells(est._clone, env, action, K, seeds)
            np.random.set_state(real)
            pi, pj = env.current_pacman_pos
            before = {g.id: (tuple(int(c) for c in g.pos), abs(g.pos[0] - pi) + abs(g.pos[1] - pj))
                      for g in env.ghosts}
        else:
            true = None
        snap = snapshot(env)
        env.step(action)
        est.observe(env, snap)
        state = agent.observe(0)
        if true is None or env.dead:
            continue
        for g in env.ghosts:
            if g.id not in true or g.id not in gen or g.id not in before:
                continue
            cells = set(neighbour_cells(env, before[g.id][0]))
            obs = tuple(int(c) for c in g.pos)
            if obs not in cells:
                continue  # eaten / respawned
            m = len(cells)
            nt, ng = sum(true[g.id].values()), sum(gen[g.id].values())
            pt = {c: (true[g.id][c] + 1) / (nt + m) for c in cells}
            pg = {c: (gen[g.id][c] + 1) / (ng + m) for c in cells}
            tv = 0.5 * sum(abs(true[g.id][c] / max(1, nt) - gen[g.id][c] / max(1, ng)) for c in cells)
            rec.append({"id": int(g.id), "dist": int(before[g.id][1]), "ll_true": math.log(pt[obs]),
                        "ll_gen": math.log(pg[obs]), "ll_unif": -math.log(m), "tv": tv})
    return rec


if __name__ == "__main__":
    ckpt, n, base, K, workers, out = sys.argv[1:7]
    n, base, K, workers = int(n), int(base), int(K), int(workers)
    ctx = multiprocessing.get_context("spawn")
    recs = []
    with ctx.Pool(workers, initializer=init, initargs=(ckpt,)) as pool:
        for r in pool.imap_unordered(play, [(g, base, K) for g in range(n)]):
            recs += r
            print(len(recs), flush=True)
    json.dump(recs, open(out, "w"))
    for name, sel in [("all", lambda r: True), ("near (<=6)", lambda r: r["dist"] <= 6)]:
        for gid in (None, 0, 1, 2):
            rs = [r for r in recs if sel(r) and (gid is None or r["id"] == gid)]
            if not rs:
                continue
            f = lambda k: np.mean([r[k] for r in rs])
            print(f"{name:11s} ghost {'all' if gid is None else gid}: n={len(rs):5d}  "
                  f"loglik true {f('ll_true'):.3f}  estimated {f('ll_gen'):.3f}  uniform {f('ll_unif'):.3f}  "
                  f"TV {f('tv'):.3f}")
