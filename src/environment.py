"""GridWorld environment with token collection mechanics."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ACTION_DELTAS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, 1),   # right
    3: (0, -1),  # left
}

# ---------------------------------------------------------------------------
# Shared color palette (RGB uint8)
# ---------------------------------------------------------------------------
_C_FLOOR  = np.array([245, 245, 245], dtype=np.uint8)  # near-white
_C_START  = np.array([200, 215, 255], dtype=np.uint8)  # pale blue
_C_GOAL   = np.array([100, 200,  80], dtype=np.uint8)  # green
_C_AGENT  = np.array([ 50, 110, 220], dtype=np.uint8)  # royal blue
_C_TOKEN  = np.array([255, 200,   0], dtype=np.uint8)  # gold (coin)
_C_GRID   = np.array([160, 160, 160], dtype=np.uint8)  # light gray (grid lines)
_C_BORDER = np.array([ 80,  80,  80], dtype=np.uint8)  # dark gray (outer border)
_C_WALL   = np.array([ 25,  25,  25], dtype=np.uint8)  # near-black (walls)


# ---------------------------------------------------------------------------
# Low-level pixel drawing helpers (vectorised, operate on H×W×3 uint8 arrays)
# ---------------------------------------------------------------------------

def _fill_circle(
    img: np.ndarray,
    cy: float, cx: float,
    radius: float,
    color: np.ndarray,
) -> None:
    """Paint a filled anti-alias-free circle onto img (in-place)."""
    r_min = max(0, int(cy - radius))
    r_max = min(img.shape[0] - 1, int(cy + radius) + 1)
    c_min = max(0, int(cx - radius))
    c_max = min(img.shape[1] - 1, int(cx + radius) + 1)
    ys = np.arange(r_min, r_max + 1)
    xs = np.arange(c_min, c_max + 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    img[r_min:r_max + 1, c_min:c_max + 1][mask] = color


def _fill_triangle(
    img: np.ndarray,
    v0: tuple[int, int],
    v1: tuple[int, int],
    v2: tuple[int, int],
    color: np.ndarray,
) -> None:
    """Paint a filled triangle defined by three (row, col) vertices (in-place).

    Uses the edge-function (cross-product) rasterisation method; all points
    on the interior and boundary are filled.
    """
    r_min = max(0, min(v0[0], v1[0], v2[0]))
    r_max = min(img.shape[0] - 1, max(v0[0], v1[0], v2[0]))
    c_min = max(0, min(v0[1], v1[1], v2[1]))
    c_max = min(img.shape[1] - 1, max(v0[1], v1[1], v2[1]))
    if r_min > r_max or c_min > c_max:
        return
    ys = np.arange(r_min, r_max + 1)
    xs = np.arange(c_min, c_max + 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    def _edge(ay, ax, by, bx):
        return (xx - ax) * (by - ay) - (yy - ay) * (bx - ax)

    e0 = _edge(v0[0], v0[1], v1[0], v1[1])
    e1 = _edge(v1[0], v1[1], v2[0], v2[1])
    e2 = _edge(v2[0], v2[1], v0[0], v0[1])
    mask = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
    img[r_min:r_max + 1, c_min:c_max + 1][mask] = color


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
        cell_px: int = 48,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n = size if size is not None else (n if n is not None else 10)
        self.max_steps = max_steps if max_steps is not None else 4 * self.n * self.n
        self.step_penalty = step_penalty
        self.collect_reward = collect_reward
        self.goal_reward = goal_reward
        self.goal_without_token_reward = goal_without_token_reward
        self._cell_px = cell_px

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
        self._window: dict | None = None  # lazy matplotlib window state

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def _get_random_token_pos(self) -> tuple[int, int]:
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
        return self._get_obs(), {}

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
            reward += self.goal_reward if self._collected else self.goal_without_token_reward

        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        info = {"reached_goal_with_token": self._collected and terminated}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering — override _render_ansi / _render_rgb_array in subclasses
    # ------------------------------------------------------------------

    def _render_ansi(self) -> str:
        """Character-grid text render."""
        n = self.n
        grid = np.full((n, n), ".", dtype=str)
        grid[n - 1, n - 1] = "G"
        if not self._collected:
            grid[self._token_pos[0], self._token_pos[1]] = "T"
        grid[self._agent_pos[0], self._agent_pos[1]] = "A"
        return "\n".join("".join(row) for row in grid)

    def _background_color(self, r: int, c: int) -> np.ndarray:
        """Return the cell background color (no entity overlay)."""
        pos = (r, c)
        if pos == (self.n - 1, self.n - 1):
            return _C_GOAL
        if pos == (0, 0):
            return _C_START
        return _C_FLOOR

    def _draw_entities(self, img: np.ndarray) -> None:
        """
        Draw agent and token as shapes on top of the background.

        - Token: filled gold circle, ~65 % of cell diameter.
        - Agent: filled blue upward-pointing equilateral triangle, ~80 % of cell.
        """
        px = self._cell_px

        def _cell_center(r: int, c: int) -> tuple[float, float]:
            return r * px + px / 2, c * px + px / 2

        # --- Token (gold coin) ---
        if not self._collected:
            tr, tc = self._token_pos
            cy, cx = _cell_center(tr, tc)
            radius = (px - 2) * 0.325        # ~65 % diameter
            _fill_circle(img, cy, cx, radius, _C_TOKEN)

        # --- Agent (upward-pointing triangle) ---
        ar, ac = self._agent_pos
        cy, cx = _cell_center(ar, ac)
        margin = max(2, (px - 2) // 10)
        half_w = (px - 2) * 0.42            # half-width of triangle base
        height  = (px - 2) * 0.80           # full height
        top    = (int(cy - height * 0.60), int(cx))           # apex
        bl     = (int(cy + height * 0.40), int(cx - half_w))  # bottom-left
        br     = (int(cy + height * 0.40), int(cx + half_w))  # bottom-right
        _fill_triangle(img, top, bl, br, _C_AGENT)

    def _render_rgb_array(self) -> np.ndarray:
        """
        Render the grid as an RGB image.

        Layout:
        - Each cell is cell_px × cell_px pixels.
        - 1-pixel grid lines between cells (_C_GRID).
        - 2-pixel outer border (_C_BORDER).
        - Agent drawn as a blue triangle; token as a gold circle.

        Total image size: (n * cell_px + 1) × (n * cell_px + 1) × 3.
        """
        n = self.n
        px = self._cell_px
        H = W = n * px + 1
        img = np.full((H, W, 3), _C_GRID, dtype=np.uint8)

        for r in range(n):
            for c in range(n):
                r0, r1 = r * px + 1, (r + 1) * px
                c0, c1 = c * px + 1, (c + 1) * px
                img[r0:r1, c0:c1] = self._background_color(r, c)

        self._draw_entities(img)

        # Outer border
        img[0, :] = _C_BORDER
        img[-1, :] = _C_BORDER
        img[:, 0] = _C_BORDER
        img[:, -1] = _C_BORDER

        return img

    def _show_image(self, img: np.ndarray) -> None:
        """Display an RGB image in a persistent, non-blocking matplotlib window."""
        import matplotlib.pyplot as plt
        if self._window is None:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.axis("off")
            fig.tight_layout(pad=0)
            im_obj = ax.imshow(img)
            self._window = {"fig": fig, "ax": ax, "im": im_obj}
        else:
            self._window["im"].set_data(img)
            self._window["fig"].canvas.flush_events()
        plt.pause(0.01)

    def render(self) -> str | np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            self._show_image(self._render_rgb_array())
            return None
        return None

    def close(self) -> None:
        if self._window is not None:
            import matplotlib.pyplot as plt
            plt.close(self._window["fig"])
            self._window = None


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
        cell_px: int = 48,
        render_mode: str | None = None,
    ):
        super().__init__(
            n=n,
            max_steps=max_steps,
            step_penalty=step_penalty,
            collect_reward=collect_reward,
            goal_reward=goal_reward,
            goal_without_token_reward=goal_without_token_reward,
            cell_px=cell_px,
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

        all_edges: list[tuple] = []
        for r in range(n):
            for c in range(n):
                if r + 1 < n:
                    all_edges.append(((r, c), (r + 1, c)))
                if c + 1 < n:
                    all_edges.append(((r, c), (r, c + 1)))

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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_ansi(self) -> str:
        """
        Maze-style ASCII render.

        Example (4x4):
            +--+--+--+--+
            |A    |     |
            +  +--+  +--+
            |  |T       |
            +  +  +  +--+
            |     |     |
            +  +--+  +  +
            |        |G |
            +--+--+--+--+
        """
        n = self.n

        def cell_char(r: int, c: int) -> str:
            pos = (r, c)
            if pos == self._agent_pos:
                return "A"
            if not self._collected and pos == self._token_pos:
                return "T"
            if pos == (n - 1, n - 1):
                return "G"
            return "."

        lines: list[str] = []
        for r in range(n):
            sep = "+"
            for c in range(n):
                above = r == 0 or self._wall_between((r - 1, c), (r, c))
                sep += ("--" if above else "  ") + "+"
            lines.append(sep)

            row = ""
            for c in range(n):
                left = c == 0 or self._wall_between((r, c - 1), (r, c))
                row += ("|" if left else " ") + cell_char(r, c) + " "
            lines.append(row + "|")

        lines.append("+" + "--+" * n)
        return "\n".join(lines)

    def _render_rgb_array(self) -> np.ndarray:
        """
        Image render: cells drawn by base class, then walls painted on top
        as thick dark lines over the shared grid-line pixels.

        Wall thickness scales with cell_px (minimum 2 px, maximum 6 px).
        """
        img = super()._render_rgb_array()
        n = self.n
        px = self._cell_px
        # Thickness: odd number so the line is centred on the grid pixel
        half = max(1, px // 16)  # half-thickness (1–3 px each side)

        for r in range(n):
            for c in range(n):
                # Right wall: vertical line at column (c+1)*px
                if c + 1 < n and self._wall_between((r, c), (r, c + 1)):
                    x = (c + 1) * px
                    r0, r1 = r * px + 1, (r + 1) * px
                    c0 = max(1, x - half)
                    c1 = min(img.shape[1] - 1, x + half + 1)
                    img[r0:r1, c0:c1] = _C_WALL
                # Bottom wall: horizontal line at row (r+1)*px
                if r + 1 < n and self._wall_between((r, c), (r + 1, c)):
                    y = (r + 1) * px
                    c0, c1 = c * px + 1, (c + 1) * px
                    r0 = max(1, y - half)
                    r1 = min(img.shape[0] - 1, y + half + 1)
                    img[r0:r1, c0:c1] = _C_WALL

        return img


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
