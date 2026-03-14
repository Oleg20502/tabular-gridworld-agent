#!/usr/bin/env python3
"""Training script for tabular Q-learning on GridWorld."""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from src.environment import GridWorldEnv
from src.q_learning import QLearningAgent
from src.state_utils import num_states


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-learning agent on GridWorld")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]

    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

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

    n_states = num_states(env_cfg["n"])
    n_episodes = train_cfg["n_episodes"]
    print(f"Training on {env_cfg['n']}x{env_cfg['n']} grid | {n_states} states | {n_episodes} episodes")
    print(f"alpha={agent_cfg['alpha']}, gamma={agent_cfg['gamma']}, epsilon={agent_cfg['epsilon']}\n")

    logs = agent.train(
        env,
        num_episodes=n_episodes,
        seed=train_cfg.get("seed"),
        max_steps_per_episode=env_cfg.get("max_steps"),
        log_interval=train_cfg["log_interval"],
    )

    agent.save(save_dir / "q_table.npy")
    shutil.copy(args.config, save_dir / "config.yaml")

    with open(save_dir / "train_logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    print(f"Saved to {save_dir.absolute()}")


if __name__ == "__main__":
    main()
