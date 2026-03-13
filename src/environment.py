"""GridWorld environment with token collection mechanics."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Action mapping: 0=up, 1=down, 2=right, 3=left
# Up decreases row index, left decreases col index
ACTION_DELTAS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, 1),   # right
    3: (0, -1),  # left
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
        step_penalty: float = -0.01,
        collect_reward: float = 1.0,
        goal_reward: float = 10.0,
        goal_without_token_penalty: float = 0.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n = size if size is not None else (n if n is not None else 10)
        self.max_steps = max_steps if max_steps is not None else 4 * self.n * self.n
        self.step_penalty = step_penalty
        self.collect_reward = collect_reward
        self.goal_reward = goal_reward
        self.goal_without_token_penalty = goal_without_token_penalty

        self.action_space = spaces.Discrete(4)
        # Observation: (agent_x, agent_y, token_x, token_y, collected)
        # When collected=1, token_x and token_y are dummy (0)
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

        # Check token collection
        if not self._collected and self._agent_pos == self._token_pos:
            self._collected = True
            reward += self.collect_reward

        # Check goal
        if self._agent_pos == (self.n - 1, self.n - 1):
            terminated = True
            if self._collected:
                reward += self.goal_reward
            else:
                reward += self.goal_without_token_penalty

        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {}
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
