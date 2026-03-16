#!/usr/bin/env python3
"""Build and save a GIF of a trained agent playing in its environment."""

import argparse
from pathlib import Path

import yaml

from src.environment import make_env
from src.q_learning import QLearningAgent
from src.visualization import make_gif


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a trained agent as a looping GIF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Run directory containing q_table.npy and config.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF path. Defaults to <save-dir>/rollout.gif",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=6,
        help="Number of episodes to include (each with a different token position)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second",
    )
    parser.add_argument(
        "--pause-frames",
        type=int,
        default=4,
        help="Extra frames to hold on the final state of each episode",
    )
    parser.add_argument(
        "--cell-px",
        type=int,
        default=56,
        help="Pixels per cell (overrides the env default for the GIF only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; episode i uses seed+i. Controls token spawn positions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)

    with open(save_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]

    env = make_env(env_cfg)

    agent = QLearningAgent(
        n=env_cfg["n"],
        alpha=agent_cfg["alpha"],
        gamma=agent_cfg["gamma"],
        exploration=agent_cfg.get("exploration", "epsilon_greedy"),
        epsilon=agent_cfg["epsilon"],
        epsilon_decay=agent_cfg["epsilon_decay"],
        min_epsilon=agent_cfg["min_epsilon"],
        temperature=agent_cfg.get("temperature", 1.0),
        temperature_decay=agent_cfg.get("temperature_decay", 1.0),
        min_temperature=agent_cfg.get("min_temperature", 0.1),
    )
    agent.load(save_dir / "q_table.npy")

    output = Path(args.output) if args.output else save_dir / "rollout.gif"

    print(
        f"Rendering {args.n_episodes} episodes "
        f"({env_cfg.get('type', 'GridWorldEnv')}, n={env_cfg['n']}) …"
    )

    gif_path = make_gif(
        env,
        agent,
        path=output,
        n_episodes=args.n_episodes,
        fps=args.fps,
        pause_frames=args.pause_frames,
        cell_px=args.cell_px,
        seed=args.seed,
    )

    print(f"Saved → {gif_path.absolute()}")


if __name__ == "__main__":
    main()
