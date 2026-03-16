"""Visualization utilities for GridWorld environments."""

from pathlib import Path


def make_gif(
    env,
    agent,
    path: str | Path,
    n_episodes: int = 5,
    fps: int = 6,
    pause_frames: int = 4,
    cell_px: int | None = None,
    seed: int | None = None,
) -> Path:
    """
    Run ``n_episodes`` with the agent's greedy policy and save a looping GIF.

    Each episode starts from a fresh ``env.reset()``, so the token spawns at
    a different position each time — useful for demonstrating generalisation.
    The GIF loops indefinitely (GIF ``loop=0``).

    Args:
        env:          A GridWorldEnv (or GridWorldWithWallsEnv) instance.
        agent:        Any agent with ``sample_action(obs, training=False)``.
        path:         Output file path, e.g. ``"runs/exp1/rollout.gif"``.
        n_episodes:   Number of episodes to include in the animation.
        fps:          Frames per second.
        pause_frames: Extra frames to hold on the final state of each episode,
                      giving the viewer time to see the outcome before the next
                      episode begins.
        cell_px:      Override the env's pixels-per-cell for the GIF only.
                      ``None`` keeps the env's current value.
        seed:         Base seed; episode i uses ``seed + i``.
                      ``None`` = non-deterministic token positions.

    Returns:
        ``Path`` to the saved GIF file.

    Raises:
        ImportError: if Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to save GIFs.  Install it with: pip install Pillow"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_cell_px = env._cell_px
    if cell_px is not None:
        env._cell_px = cell_px

    try:
        frames: list[Image.Image] = []

        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            obs, _ = env.reset(seed=ep_seed)
            frames.append(Image.fromarray(env._render_rgb_array()))

            for _ in range(env.max_steps):
                action = agent.sample_action(obs, training=False)
                obs, _reward, terminated, truncated, _ = env.step(action)
                frames.append(Image.fromarray(env._render_rgb_array()))
                if terminated or truncated:
                    break

            # Hold the final frame so the viewer sees the outcome
            for _ in range(pause_frames):
                frames.append(frames[-1].copy())

    finally:
        env._cell_px = original_cell_px

    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,          # 0 = loop forever
        duration=int(1000 / fps),
        optimize=False,  # skip palette re-quantization; preserves exact colors
    )

    return path
