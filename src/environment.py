"""GridWorld environment with token collection mechanics."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, 1),
    3: (0, -1),
}


class GridWorldEnv(gym.Env):
    """
    GridWorld environment: agent collects token and reaches goal.
    - Grid NxN (default N=10)
    - Agent starts at (0, 0), goal at (N-1, N-1)
    - Token spawns randomly except (0,0) and (N-1,N-1)
    - Agent gets final reward only if token was collected first.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        n: int | None = None,
        size: int | None = None,
        max_steps: int | None = None,
        step_penalty: float = -1.0,
        collect_reward: float = 10.0,
        goal_reward: float = 20.0,
        goal_without_token_reward: float = 0.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n = size if size is not None else (n if n is not None else 10)
        self.max_steps = max_steps if max_steps is not None else 4 * self.n * self.n
        self.step_penalty = step_penalty
        self.collect_reward = collect_reward
        self.goal_reward = goal_reward
        self.goal_without_token_reward = goal_without_token_reward

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(self.n),
                spaces.Discrete(self.n),
                spaces.Discrete(self.n),
                spaces.Discrete(self.n),
                spaces.Discrete(2),
            )
        )

        self.render_mode = render_mode
        self._agent_pos = (0, 0)
        self._token_pos = (0, 0)
        self._collected = False
        self._step_count = 0

    def _get_random_token_pos(self) -> tuple[int, int]:
        """Sample token position excluding (0,0) and (N-1, N-1)."""
        excluded = {(0, 0), (self.n - 1, self.n - 1)}
        candidates = [
            (r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in excluded
        ]
        idx = self.np_random.integers(0, len(candidates))
        return candidates[idx]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[tuple[int, int, int, int, int], dict]:
        super().reset(seed=seed)
        self._agent_pos = (0, 0)
        self._token_pos = self._get_random_token_pos()
        self._collected = False
        self._step_count = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> tuple[int, int, int, int, int]:
        ax, ay = self._agent_pos
        if self._collected:
            return (ax, ay, 0, 0, 1)
        tx, ty = self._token_pos
        return (ax, ay, tx, ty, 0)

    def step(
        self, action: int
    ) -> tuple[tuple[int, int, int, int, int], float, bool, bool, dict]:
        ax, ay = self._agent_pos
        dx, dy = ACTION_DELTAS[action]
        new_ax = np.clip(ax + dx, 0, self.n - 1)
        new_ay = np.clip(ay + dy, 0, self.n - 1)
        self._agent_pos = (int(new_ax), int(new_ay))
        self._step_count += 1

        reward = self.step_penalty
        terminated = False
        truncated = False

        if not self._collected and self._agent_pos == self._token_pos:
            self._collected = True
            reward += self.collect_reward

        if self._agent_pos == (self.n - 1, self.n - 1):
            terminated = True
            if self._collected:
                reward += self.goal_reward
            else:
                reward += self.goal_without_token_reward

        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {"reached_goal_with_token": self._collected and terminated}
        return obs, reward, terminated, truncated, info

    def render(self) -> str | np.ndarray | None:
        if self.render_mode is None:
            return None
        grid = np.full((self.n, self.n), ".", dtype=str)
        grid[self._agent_pos[0], self._agent_pos[1]] = "A"
        if not self._collected:
            grid[self._token_pos[0], self._token_pos[1]] = "T"
        grid[0, 0] = "S" if self._agent_pos == (0, 0) else grid[0, 0]
        grid[self.n - 1, self.n - 1] = "G" if self._agent_pos != (self.n - 1, self.n - 1) else "A"
        lines = ["".join(row) for row in grid]
        out = "\n".join(lines)
        if self.render_mode == "ansi":
            return out
        if self.render_mode == "human":
            print(out)
            return None
        if self.render_mode == "rgb_array":
            # Simple text-based rgb_array: render as ASCII to small image
            return np.array([])  # Minimal implementation
        return None

    def close(self) -> None:
        pass


class GridWorldWithWallsEnv(GridWorldEnv):
    """
    GridWorld with static walls between cells.

    Walls are generated once at construction and persist across episodes.
    Full connectivity of all cells is guaranteed by construction: a random
    spanning tree is built first (ensuring every cell is reachable), then
    additional walls are added only on non-spanning-tree edges.
    """

    def __init__(
        self,
        n: int = 10,
        wall_frac: float = 0.05,
        seed_walls: int | None = None,
        max_steps: int | None = None,
        step_penalty: float = -1.0,
        collect_reward: float = 10.0,
        goal_reward: float = 20.0,
        goal_without_token_reward: float = 0.0,
        render_mode: str | None = None,
    ):
        super().__init__(
            n=n,
            max_steps=max_steps,
            step_penalty=step_penalty,
            collect_reward=collect_reward,
            goal_reward=goal_reward,
            goal_without_token_reward=goal_without_token_reward,
            render_mode=render_mode,
        )
        rng = np.random.default_rng(seed_walls)
        self._walls: frozenset = self._generate_walls(wall_frac, rng)

    def _generate_walls(self, wall_frac: float, rng: np.random.Generator) -> frozenset:
        """
        Generate walls while guaranteeing full cell connectivity.

        Strategy:
        1. Build a random spanning tree via iterative DFS — these edges can never
           become walls, ensuring every cell is reachable from every other.
        2. From the remaining (non-tree) edges, randomly select
           floor(wall_frac * total_edges) to become walls.
        """
        n = self.n

        # All possible undirected edges between adjacent cells
        all_edges: list[tuple] = []
        for r in range(n):
            for c in range(n):
                if r + 1 < n:
                    all_edges.append(((r, c), (r + 1, c)))
                if c + 1 < n:
                    all_edges.append(((r, c), (r, c + 1)))

        # Iterative randomized DFS to build spanning tree
        visited: set = {(0, 0)}
        tree_edges: set = set()
        stack: list = [(0, 0)]

        while stack:
            r, c = stack[-1]
            unvisited = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < n and 0 <= c + dc < n
                and (r + dr, c + dc) not in visited
            ]
            if unvisited:
                nb = unvisited[rng.integers(len(unvisited))]
                visited.add(nb)
                tree_edges.add((min((r, c), nb), max((r, c), nb)))
                stack.append(nb)
            else:
                stack.pop()

        # Non-spanning-tree edges are candidates for walls
        non_tree = [e for e in all_edges if e not in tree_edges]
        n_walls = min(int(wall_frac * len(all_edges)), len(non_tree))

        if n_walls > 0:
            chosen = rng.choice(len(non_tree), size=n_walls, replace=False)
            return frozenset(non_tree[i] for i in chosen)
        return frozenset()

    def _wall_between(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        return (min(a, b), max(a, b)) in self._walls

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(self, action: int):
        ax, ay = self._agent_pos
        dx, dy = ACTION_DELTAS[action]
        new_pos = (int(np.clip(ax + dx, 0, self.n - 1)),
                   int(np.clip(ay + dy, 0, self.n - 1)))

        # Move only when not blocked by a wall
        if new_pos != self._agent_pos and not self._wall_between(self._agent_pos, new_pos):
            self._agent_pos = new_pos

        self._step_count += 1
        reward = self.step_penalty
        terminated = False
        truncated = False

        if not self._collected and self._agent_pos == self._token_pos:
            self._collected = True
            reward += self.collect_reward

        if self._agent_pos == (self.n - 1, self.n - 1):
            terminated = True
            reward += self.goal_reward if self._collected else self.goal_without_token_reward

        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        info = {"reached_goal_with_token": self._collected and terminated}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> str | None:
        """
        Maze-style ASCII render.

        Each cell is 3 chars wide: separator + content + space.
        Horizontal walls shown as '--', absent as '  '.
        Vertical walls shown as '|', absent as ' '.

        Example (4x4, walls between some cells):
            +--+--+--+--+
            |S    |  |  |
            +  +--+  +--+
            |  |T |     |
            +--+  +--+  +
            |     |  |  |
            +  +--+  +--+
            |  |  |  |G |
            +--+--+--+--+
        """
        if self.render_mode is None:
            return None

        n = self.n

        def cell_char(r: int, c: int) -> str:
            pos = (r, c)
            if pos == self._agent_pos:
                return "A"
            if not self._collected and pos == self._token_pos:
                return "T"
            if pos == (n - 1, n - 1):
                return "G"
            if pos == (0, 0):
                return "S"
            return "."

        lines: list[str] = []
        for r in range(n):
            # Separator row above row r
            sep = "+"
            for c in range(n):
                above_wall = r == 0 or self._wall_between((r - 1, c), (r, c))
                sep += ("--" if above_wall else "  ") + "+"
            lines.append(sep)

            # Content row
            row = ""
            for c in range(n):
                left_wall = c == 0 or self._wall_between((r, c - 1), (r, c))
                row += ("|" if left_wall else " ") + cell_char(r, c) + " "
            row += "|"
            lines.append(row)

        lines.append("+" + "--+" * n)

        out = "\n".join(lines)
        if self.render_mode == "ansi":
            return out
        if self.render_mode == "human":
            print(out)
            return None
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENV_REGISTRY: dict[str, type] = {
    "GridWorldEnv": GridWorldEnv,
    "GridWorldWithWallsEnv": GridWorldWithWallsEnv,
}


def make_env(env_cfg: dict, render_mode: str | None = None) -> gym.Env:
    """Instantiate an environment from a config dict.

    The dict may contain a ``type`` key (default: ``"GridWorldEnv"``) and any
    constructor kwargs for that class.  ``type`` is stripped before forwarding.
    """
    cfg = dict(env_cfg)
    env_type = cfg.pop("type", "GridWorldEnv")
    if env_type not in _ENV_REGISTRY:
        raise ValueError(f"Unknown env type '{env_type}'. Choose from: {list(_ENV_REGISTRY)}")
    if render_mode is not None:
        cfg["render_mode"] = render_mode
    return _ENV_REGISTRY[env_type](**cfg)
