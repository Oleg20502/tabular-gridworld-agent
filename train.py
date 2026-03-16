#!/usr/bin/env python3
"""Training script for tabular RL agents on GridWorld.

Supported algorithms (set via ``agent.algorithm`` in the config):
  - q_learning   : Off-policy Q-learning (default)
  - sarsa        : On-policy SARSA (TD(0))
  - monte_carlo  : On-policy first-visit Monte Carlo control
  - q_lambda     : Watkins's Q(λ) with eligibility traces (extra param: lam)
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from src.environment import make_env
from src.q_learning import QLearningAgent
from src.sarsa import SARSAAgent
from src.monte_carlo import MonteCarloAgent
from src.q_lambda import QLambdaAgent
from src.state_utils import num_states


AGENT_CLASSES = {
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "monte_carlo": MonteCarloAgent,
    "q_lambda": QLambdaAgent,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tabular RL agent on GridWorld")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Seed for evaluation")
    return parser.parse_args()


def build_agent(env_cfg: dict, agent_cfg: dict):
    algorithm = agent_cfg.get("algorithm", "q_learning")
    if algorithm not in AGENT_CLASSES:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Choose from: {list(AGENT_CLASSES)}"
        )

    kwargs = dict(
        n=env_cfg["n"],
        alpha=agent_cfg["alpha"],
        gamma=agent_cfg["gamma"],
        exploration=agent_cfg.get("exploration", "epsilon_greedy"),
        epsilon=agent_cfg.get("epsilon", 0.1),
        epsilon_decay=agent_cfg.get("epsilon_decay", 1.0),
        min_epsilon=agent_cfg.get("min_epsilon", 0.01),
        temperature=agent_cfg.get("temperature", 1.0),
        temperature_decay=agent_cfg.get("temperature_decay", 1.0),
        min_temperature=agent_cfg.get("min_temperature", 0.1),
    )

    if algorithm == "q_lambda":
        kwargs["lam"] = agent_cfg.get("lam", 0.9)

    return AGENT_CLASSES[algorithm](**kwargs)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]

    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, save_dir / "config.yaml")

    env = make_env(env_cfg)
    agent = build_agent(env_cfg, agent_cfg)

    algorithm = agent_cfg.get("algorithm", "q_learning")
    n_states = num_states(env_cfg["n"])
    n_episodes = train_cfg["n_episodes"]
    exploration = agent_cfg.get("exploration", "epsilon_greedy")
    print(
        f"Algorithm: {algorithm} | "
        f"{env_cfg['n']}x{env_cfg['n']} grid | {n_states} states | {n_episodes} episodes"
    )
    print(f"alpha={agent_cfg['alpha']}, gamma={agent_cfg['gamma']}, exploration={exploration}\n")

    logs = agent.train(
        env,
        num_episodes=n_episodes,
        seed=train_cfg.get("seed"),
        max_steps_per_episode=env_cfg.get("max_steps"),
        log_interval=train_cfg["log_interval"],
    )

    agent.save(save_dir / "q_table.npy")

    with open(save_dir / "train_logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    print(f"Saved to {save_dir.absolute()}")


    # Evaluation
    env = make_env(env_cfg)

    agent = build_agent(env_cfg, agent_cfg)
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
