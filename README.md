# Tabular Q-Learning for GridWorld

A clean Python implementation of four classic tabular reinforcement learning algorithms — **Q-Learning**, **SARSA**, **Monte Carlo Control**, and **Q(λ)** — trained and evaluated on a custom GridWorld environment built with the [Gymnasium](https://gymnasium.farama.org/) API.

---

## Environments

### GridWorld

<p align="center">
  <img src="assets/gridworld.png" width="300"/>
</p>

An agent navigates an N×N grid, collects a token, and reaches the goal. The reward is only given for reaching the goal *after* collecting the token, which requires the agent to plan a two-stage path.

| Property | Value |
|---|---|
| Grid size | N×N (default N = 10) |
| Agent start | top-left corner (0, 0) |
| Goal | bottom-right corner (N−1, N−1) |
| Token | random position ≠ (0,0) and ≠ (N−1,N−1) at episode start |
| Actions | `{up, down, right, left}` — boundary moves are no-ops |
| Episode end | agent reaches (N−1,N−1) **or** step limit exceeded (default 4N²) |

**Rewards**

| Event | Default reward |
|---|---|
| Each step | −1 |
| Collecting token | +10 |
| Reaching goal *with* token | +10 |
| Reaching goal *without* token | 0 |

The large step penalty makes the agent learn short paths. Without it, any path that eventually reaches the goal has the same total reward, so the agent has no incentive to improve.

---

### GridWorld with Walls

<p align="center">
  <img src="assets/gridworldwithwalls.png" width="300"/>
</p>

An extension of GridWorld where **static walls** are placed between cells. Walls are generated once at construction time and remain fixed for all episodes. A wall between two adjacent cells prevents movement across that border.

| Property | Value |
|---|---|
| `wall_frac` | fraction of all edges that become walls |
| `seed_walls` | fixes the wall layout for reproducibility |
| Maximum `wall_frac` | (N−1) / 2N ≈ 45% for N = 10 |

**Wall generation** is guaranteed to preserve full connectivity — the agent can always reach the token and the goal regardless of `wall_frac`. This is achieved by:

1. Building a random **spanning tree** of the grid via iterative randomised DFS. These N²−1 edges are protected — they can never become walls.
2. Randomly assigning walls to at most (N−1)² of the remaining non-tree edges.

---

### State Space

The full observation is a 5-tuple:

$$s = (a_x,\ a_y,\ t_x,\ t_y,\ c) \quad \text{where}\ c \in \{0, 1\}$$

- $(a_x, a_y)$: agent position
- $(t_x, t_y)$: token position
- $c$: whether the token has been collected

**State space reduction.** Once the token is collected ($c = 1$), its position is irrelevant. The naïve state space has size $2N^4$, but we reduce it to $N^4 + N^2$ by using two disjoint index ranges:

$$\text{index}(s) = \begin{cases} a_x N^3 + a_y N^2 + t_x N + t_y & \text{if } c = 0 \quad (\text{range } [0,\ N^4)) \\ N^4 + a_x N + a_y & \text{if } c = 1 \quad (\text{range } [N^4,\ N^4 + N^2)) \end{cases}$$

For N = 10 this gives **10,100 states** instead of 20,000 — nearly a 2× reduction.

---

## Methods

All agents maintain a tabular **Q-function** $Q: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ stored as a matrix of shape $|\mathcal{S}| \times |\mathcal{A}|$, initialised to zero.

---

### Q-Learning (off-policy TD)

Q-Learning learns the optimal Q-function directly, independently of the behaviour policy, by bootstrapping from the **maximum** Q-value in the next state.

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Because the target uses $\max_{a'}$, the algorithm is **off-policy**: it converges to $Q^*$ regardless of how actions are chosen during training.

```
Algorithm: Q-Learning
─────────────────────────────────────────────────────────
Initialise Q(s, a) = 0 for all s, a
For each episode:
    s ← env.reset()
    For each step:
        a ← ε-greedy(Q, s) # or softmax
        s', r, done ← env.step(a)
        if done:
            target = r
        else:
            target = r + γ max_a' Q(s', a')
        Q(s, a) ← Q(s, a) + α (target − Q(s, a))
        s ← s'
        if done: break
    ε ← max(ε_min, ε · ε_decay)
```

---

### SARSA (on-policy TD)

SARSA bootstraps from the Q-value of the **action actually taken** in the next state $a'$, making it on-policy. The policy used to generate $(s, a, r, s', a')$ is the same policy being improved.

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma\, Q(s', a') - Q(s, a) \right]$$

Because $a'$ is sampled from the current (exploratory) policy, SARSA is more conservative than Q-Learning — it accounts for the fact that the agent will continue to explore in $s'$.

```
Algorithm: SARSA
─────────────────────────────────────────────────────────
Initialise Q(s, a) = 0 for all s, a
For each episode:
    s ← env.reset()
    a ← ε-greedy(Q, s)
    For each step:
        s', r, done ← env.step(a)
        a' ← ε-greedy(Q, s')
        if done:
            target = r
        else:
            target = r + γ · Q(s', a')
        Q(s, a) ← Q(s, a) + α · (target − Q(s, a))
        s ← s',  a ← a'
        if done: break
    ε ← max(ε_min, ε · ε_decay)
```

---

### First-Visit Monte Carlo Control

Monte Carlo control waits for the **complete episode** to finish, then updates every first-visited $(s, a)$ pair using the actual discounted return $G_t$:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$$

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ G_t - Q(s_t, a_t) \right]$$

Unlike TD methods, MC requires no bootstrapping and has zero bias (returns are exact), but has higher variance and cannot learn from incomplete episodes.

```
Algorithm: First-Visit Monte Carlo Control
─────────────────────────────────────────────────────────
Initialise Q(s, a) = 0 for all s, a
For each episode:
    s ← env.reset()
    trajectory ← []
    Collect full episode:
        a ← ε-greedy(Q, s)
        s', r, done ← env.step(a)
        trajectory.append((s, a, r))
        s ← s'
        if done: break
    G ← 0
    visited ← {}
    For (s, a, r) in reversed(trajectory):
        G ← r + γ · G
        if (s, a) not in visited:
            visited.add((s, a))
            Q(s, a) ← Q(s, a) + α · (G − Q(s, a))
    ε ← max(ε_min, ε · ε_decay)
```

---

### Q(λ) — Watkins's Q with Eligibility Traces

Q(λ) generalises Q-Learning by maintaining **eligibility traces** $e(s, a)$ that accumulate credit for recently visited state–action pairs. This allows TD errors to propagate backwards through the trajectory, combining the sample efficiency of TD with the multi-step credit assignment of Monte Carlo.

The key parameter $\lambda \in [0, 1]$ controls the trade-off:
- $\lambda = 0$: reduces to one-step Q-Learning
- $\lambda = 1$: approaches full Monte Carlo returns (while remaining off-policy)

**Watkins's cut:** When an exploratory (non-greedy) action is taken, traces are reset to zero. This preserves the off-policy convergence guarantee by preventing credit from propagating through exploratory transitions.

$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$

$$e(s_t, a_t) \mathrel{+}= 1 \quad \text{(accumulating trace)}$$

$$Q(s, a) \leftarrow Q(s, a) + \alpha\, \delta_t\, e(s, a) \quad \forall\, s, a$$

$$e(s, a) \leftarrow \begin{cases} \gamma\lambda\, e(s, a) & \text{if greedy action was taken} \\ 0 & \text{if exploratory action (Watkins's cut)} \end{cases}$$

```
Algorithm: Watkins's Q(λ)
─────────────────────────────────────────────────────────
Initialise Q(s, a) = 0,  e(s, a) = 0  for all s, a
For each episode:
    e ← 0  (reset all traces)
    s ← env.reset()
    For each step:
        a ← ε-greedy(Q, s)
        s', r, done ← env.step(a)
        a* ← argmax_a' Q(s', a')
        δ ← r + γ · Q(s', a*) − Q(s, a)
        e(s, a) ← e(s, a) + 1
        Q ← Q + α · δ · e
        if done:
            e ← 0
        elif a == a*:
            e ← γλ · e
        else:
            e ← 0
        s ← s'
        if done: break
    ε ← max(ε_min, ε · ε_decay)
```

---

## Exploration Strategies

All agents support two exploration strategies, switchable via the config.

### ε-Greedy

With probability $\varepsilon$ the agent selects a random action; otherwise it acts greedily:

$$a = \begin{cases} \text{random} & \text{with probability } \varepsilon \\ \arg\max_{a'} Q(s, a') & \text{with probability } 1 - \varepsilon \end{cases}$$

$\varepsilon$ decays multiplicatively each episode: $\varepsilon \leftarrow \max(\varepsilon_{\min},\ \varepsilon \cdot d_\varepsilon)$.

### Softmax with temperature

All actions are sampled proportionally to their exponentiated Q-values, controlled by a temperature $\tau > 0$:

$$\pi(a \mid s) = \frac{\exp\!\left(Q(s, a)\,/\,\tau\right)}{\sum_{a'} \exp\!\left(Q(s, a')\,/\,\tau\right)}$$

- High $\tau$: near-uniform distribution (more exploration)
- Low $\tau$: distribution concentrates on the greedy action

$\tau$ decays multiplicatively each episode: $\tau \leftarrow \max(\tau_{\min},\ \tau \cdot d_\tau)$.


---

## Experiments

Experiments were run on 10×10 GridWorld and GridWorld-with-Walls environments with 20,000 training episodes.

### Reward Shaping

The reward structure has a large impact on convergence:

- **No step penalty** (`step_penalty = 0`): in theory may provide unoptimal paths but doesn't penalize exploration
- **Step penalty** (`step_penalty = −1`): in theory should encourage short

### ε-Greedy vs. Softmax

Both strategies were tested with and without decay schedules.

| Setting | Behaviour |
|---|---|
| **Constant ε** | Fixed exploration throughout; never fully exploits. |
| **Decaying ε** (`ε_decay = 0.9999`) | Exploration tapers off; converges to near-greedy policy. |
| **Constant τ** | Temperature stays high; stochastic even in late training. |
| **Decaying τ** (`τ_decay = 0.9999`) | Similar to decaying ε but softer early-stage exploration; often finds better paths sooner because it does not waste steps on purely random actions. |

Key finding: **softmax with decay** tends to converge more smoothly than ε-greedy with decay because even during exploration it still prefers actions with higher Q-values, rather than selecting uniformly at random.

---

## Usage

### Install

```bash
pip install -r requirements.txt
```

### Configure

Use `config.yaml` to set experiment hyperparameters:

```yaml
env:
  type: GridWorldEnv          # or GridWorldWithWallsEnv
  n: 10
  wall_frac: 0.1             # only for GridWorldWithWallsEnv
  seed_walls: 0
  max_steps: 1000
  step_penalty: -1
  collect_reward: 10.0
  goal_reward: 10.0

agent:
  algorithm: q_learning       # q_learning | sarsa | monte_carlo | q_lambda
  alpha: 0.1
  gamma: 0.99
  exploration: softmax        # epsilon_greedy | softmax
  temperature: 5.0
  temperature_decay: 0.9999
  min_temperature: 0.01

train:
  save_dir: runs/myexp
  n_episodes: 20000
  log_interval: 500
  seed: 42
```

### Train

```bash
python train.py --config config.yaml
```

### Evaluate

```bash
python eval.py --save-dir runs/myexp --n-episodes 1000 --seed 0
```

