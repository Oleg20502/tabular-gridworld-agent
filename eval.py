#!/usr/bin/env python3
"""Evaluation script for tabular Q-learning on GridWorld."""

import argparse
import json
from pathlib import Path

import yaml

from src.environment import GridWorldEnv
from src.q_learning import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Q-learning agent on GridWorld")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory with saved Q-table and config")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Seed for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)

    with open(save_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]

    env = GridWorldEnv(
        n=env_cfg["n"],
        max_steps=env_cfg.get("max_steps"),
        step_penalty=env_cfg["step_penalty"],
        collect_reward=env_cfg["collect_reward"],
        goal_reward=env_cfg["goal_reward"],
        goal_without_token_reward=env_cfg["goal_without_token_reward"],
    )

    agent = QLearningAgent(
        n=env_cfg["n"],
        alpha=agent_cfg["alpha"],
        gamma=agent_cfg["gamma"],
        epsilon=agent_cfg["epsilon"],
        epsilon_decay=agent_cfg["epsilon_decay"],
        min_epsilon=agent_cfg["min_epsilon"],
    )
    agent.load(save_dir / "q_table.npy")

    print(f"Evaluating for {args.n_episodes} episodes:")

    results = agent.evaluate(env, args.n_episodes, seed=args.seed)

    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Average reward: {results['avg_reward']:.2f}")
    print(f"STD of reward: {results['std_reward']:.2f}")

    with open(save_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {(save_dir / 'eval_results.json').absolute()}")


if __name__ == "__main__":
    main()
