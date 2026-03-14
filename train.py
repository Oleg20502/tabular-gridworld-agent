#!/usr/bin/env python3
"""Training script for tabular Q-learning on GridWorld."""

import argparse
import json
from pathlib import Path

from src.environment import GridWorldEnv
from src.q_learning import QLearningAgent
from src.state_utils import num_states


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-learning agent on GridWorld")
    parser.add_argument("--n", type=int, default=10, help="Grid size NxN")
    parser.add_argument("--episodes", type=int, default=1_000, help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/",
        help="Directory to save Q-table and config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env = GridWorldEnv(size=args.n)
    n_states = num_states(args.n)
    agent = QLearningAgent(
        n=args.n,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
    )

    print(f"Training on {args.n}x{args.n} grid, {n_states} states, {args.episodes} episodes")
    print(f"alpha={args.alpha}, gamma={args.gamma}, epsilon={args.epsilon}\n")

    rewards, success_count = agent.train(
        env, num_episodes=args.episodes, seed=args.seed
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save(save_dir / "q_table.npy")
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(
            {
                "n": args.n,
                "episodes": args.episodes,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "epsilon": args.epsilon,
                "num_states": n_states,
            },
            f,
            indent=2,
        )

    print(f"\nTrained for {args.episodes} episodes")
    success_rate = success_count / args.episodes
    print(f"Training success rate: {success_rate:.2%}")
    print(f"\nCheckpoints saved to {save_dir.absolute()}")

if __name__ == "__main__":
    main()
