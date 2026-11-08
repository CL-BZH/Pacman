"""
Linear value function approximation on the reduced Pacman game of
chapter 3 (course, chapter 6). The value of the optimal policy is
approximated with a few hand-made features; we compare the TD fixed point,
the best approximation (projection) and the bound of the course.
"""
import numpy as np
import dp_escape as d

v_star, _ = d.value_iteration()
pi = d.greedy(v_star)
live = [k for k, (p, g) in enumerate(d.S) if p != g and p != d.GOAL]
pos = {k: j for j, k in enumerate(live)}
n = len(live)

# Transition matrix (sub-stochastic: terminal states are absorbing with
# value 0, hence dropped) and expected rewards of the policy pi
P = np.zeros((n, n)); r = np.zeros(n)
for k in live:
    for pr, rew, k2 in d.T[k][pi[k]]:
        r[pos[k]] += pr * rew
        if k2 >= 0 and k2 in pos:
            P[pos[k], pos[k2]] += pr * d.GAMMA
v = np.linalg.solve(np.eye(n) - P, r)          # exact v_pi on live states
assert np.allclose(v, v_star[live], atol=1e-6)

# Weighting: expected number of visits from a uniform start (undiscounted
# occupancy), which satisfies d P_pi <= d
P1 = P / d.GAMMA
mu0 = np.full(n, 1.0 / n)
occ = np.linalg.solve((np.eye(n) - P1).T, mu0)
D = occ / occ.sum()


def features(k):
    p, g = d.S[k]
    dg = d.dist[d.idx[p], d.idx[d.GOAL]]           # Pacman -> goal
    dh = d.dist[d.idx[p], d.idx[g]]                # Pacman -> ghost
    dgh = d.dist[d.idx[g], d.idx[d.GOAL]]          # ghost -> goal
    return np.array([1.0, dg, dh, dgh, dg * dg, min(dh, 6.0),
                     float(dh <= 2), float(dgh < dg)])


X = np.array([features(k) for k in live])
Dm = np.diag(D)
# Best approximation: weighted least squares (projection of v)
w_proj = np.linalg.solve(X.T @ Dm @ X, X.T @ Dm @ v)
# TD fixed point: A w = b with A = X^T D (I - gamma P) X, b = X^T D r
A = X.T @ Dm @ (np.eye(n) - P) @ X
b = X.T @ Dm @ r
w_td = np.linalg.solve(A, b)
norm = lambda e: np.sqrt(np.sum(D * e * e))
e_proj, e_td = norm(X @ w_proj - v), norm(X @ w_td - v)
print(f"{n} states, {X.shape[1]} features; ||v||_D = {norm(v):.3f}")
print(f"best approximation error ||Pi v - v||_D = {e_proj:.3f}")
print(f"TD fixed point error     ||v_TD - v||_D = {e_td:.3f}")
print(f"bound ||Pi v - v||_D / sqrt(1 - gamma^2) = "
      f"{e_proj / np.sqrt(1 - d.GAMMA ** 2):.3f}")
sym = (A + A.T) / 2
print(f"smallest eigenvalue of the symmetric part of A: "
      f"{np.linalg.eigvalsh(sym).min():.4f} (> 0)")
