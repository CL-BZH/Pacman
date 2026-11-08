# Reinforcement Learning with Pacman — the course

[`rl_with_pacman.pdf`](rl_with_pacman.pdf) is a comprehensive course on reinforcement learning,
from bandits and Markov decision processes to deep RL, planning, offline RL, multi-agent RL and
RLHF. The Pacman game of this repository is the running example: the algorithms it uses (DQN,
A2C/PPO, look-ahead search, expert iteration) are explained with excerpts of its code, and the
results quoted were obtained with it.

| Chapter | Content |
|---|---|
| 1. Introduction: Pacman as an RL problem | What RL is and where it is used; a map of the methods; the agent–environment loop; the rules, rewards and observations of games v1 and v2 |
| 2. Multi-armed bandits | Regret, ε-greedy, UCB1 (with proof), Lai–Robbins bound, Thompson sampling, gradient bandits, LinUCB |
| 3. Markov decision processes | Returns, value functions, Bellman equations, contractions, optimal policies, other criteria |
| 4. Dynamic programming | Policy and value iteration, asynchronous DP, linear programming; a reduced Pacman game solved exactly |
| 5. Monte Carlo and temporal-difference learning | MC, TD(0), n-step and λ-returns, eligibility traces, SARSA, expected SARSA, Q-learning, off-policy MC |
| 6. Value function approximation | Linear TD and its fixed point, error bound, LSTD, Baird's counterexample, gradient TD |
| 7. Deep Q-networks | Deadly triad, replay, target network, double DQN, dueling, n-step; Rainbow, distributional RL |
| 8. Policy gradients and actor-critic methods | Policy gradient theorem, baselines, GAE, A2C, Gaussian policies, natural gradient |
| 9. Trust regions: TRPO and PPO | Performance difference lemma, TRPO bound, clipped objective; results on Pacman |
| 10. Continuous actions | Deterministic policy gradient, DDPG, TD3, soft actor-critic |
| 11. Model-based RL and planning | Dyna, simulation lemma, MCTS/UCT, AlphaZero, MuZero; Pacman's look-ahead search and expert iteration |
| 12. Imitation learning and offline RL | Behaviour cloning and compounding errors, DAgger, max-ent IRL, GAIL, extrapolation error, TD3+BC, CQL, IQL |
| 13. Exploration | Why dithering fails, R-max and bonuses, pseudo-counts, RND, curiosity, Go-Explore, posterior sampling |
| 14. Multi-agent RL | Markov games, Shapley's theorem, minimax-Q, self-play, CTDE (MADDPG, MAPPO), VDN/QMIX |
| 15. RL from human feedback | Bradley–Terry reward models, KL-regularised PPO, GRPO, DPO (with derivation), RLAIF |

Proofs are given in full. Boxes are colour-coded: blue = definitions, red = theorems, amber =
propositions, green = Pacman examples, grey = remarks, violet = algorithms. Teal "In words"
boxes explain the intuition behind the main formulas.

## Files

- `rl_with_pacman.tex`: main file; `chapters/`: one file per chapter; `course.sty`: style;
  `figures/`: TikZ figures.
- `code/bandits.py`: the bandit testbed (chapter 2).
- `code/dp_escape.py`: exact dynamic programming on a reduced Pacman game (chapter 4).
- `code/td_escape.py`: Q-learning and SARSA on the same game (chapter 5).
- `code/vfa_escape.py`: linear TD and LSTD on the same game (chapter 6).

## Compiling

```bash
cd Course
latexmk -pdf rl_with_pacman.tex
```

The course uses standard LaTeX packages (`tcolorbox`, `tikz`, `algorithmicx`, `cleveref`, Latin
Modern fonts). The training curves of chapter 9 are read from `../Media/a2c_ppo_training.png`.

The scripts of `code/` run from any folder: `python Course/code/dp_escape.py`.
