#!/usr/bin/env python3
"""Evaluation script for tabular Q-learning on GridWorld."""

import argparse
import json
from pathlib import Path

from src.environment import GridWorldEnv
from src.q_learning import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Q-learning agent on GridWorld")
    parser.add_argument("--n", type=int, default=10, help="Grid size NxN")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/",
        help="Directory to load Q-table from",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)

    with open(save_dir / "config.json") as f:
        config = json.load(f)
    
    agent = QLearningAgent(n=config["n"])
    agent.load(save_dir / "q_table.npy")

    eval_env = GridWorldEnv(size=config["n"])
    
    print(f"\nEvaluating for {args.eval_episodes} episodes:")

    results = agent.evaluate(eval_env, args.eval_episodes, seed=args.seed)

    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Average reward: {results['avg_reward']:.2f}")
    print(f"STD of reward: {results['std_reward']:.2f}")

    with open(save_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
