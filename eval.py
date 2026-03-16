#!/usr/bin/env python3
"""Evaluation script for tabular Q-learning on GridWorld."""

import argparse
import json
from pathlib import Path

import yaml

from src.environment import make_env
from src.q_learning import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Q-learning agent on GridWorld")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory with saved Q-table and config")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Seed for evaluation")
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

    print(f"Evaluating for {args.n_episodes} episodes:")

    results = agent.evaluate(
        env,
        args.n_episodes,
        max_steps_per_episode=env_cfg.get("max_steps"),
        seed=args.seed,
    )

    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Average reward: {results['avg_reward']:.2f}")
    print(f"STD of reward: {results['std_reward']:.2f}")
    print(f"Average episode length: {results['avg_episode_length']:.1f}")
    print(f"STD of episode length: {results['std_episode_length']:.1f}")
    print(f"Average episode length (successful): {results['avg_success_episode_length']:.1f}")
    print(f"STD of episode length (successful): {results['std_success_episode_length']:.1f}")

    with open(save_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {(save_dir / 'eval_results.json').absolute()}")


if __name__ == "__main__":
    main()
