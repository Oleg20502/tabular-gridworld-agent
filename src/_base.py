"""Shared base class for tabular Q-function learning agents."""

from pathlib import Path

import numpy as np

from .state_utils import num_states, state_to_index


EXPLORATION_STRATEGIES = ("epsilon_greedy", "softmax")


class BaseTabularAgent:
    """Common utilities for tabular agents that learn a Q-function.

    Subclasses must implement ``train()``.
    """

    def __init__(
        self,
        n: int,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        exploration: str = "epsilon_greedy",
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
        temperature: float = 1.0,
        temperature_decay: float = 1.0,
        min_temperature: float = 0.1,
    ):
        """Initialize shared agent parameters.

        Args:
            n: Grid size (NxN). Used to compute num_states and for state_to_index.
            num_actions: Number of discrete actions (default 4).
            alpha: Learning rate.
            gamma: Discount factor.
            exploration: Exploration strategy — "epsilon_greedy" or "softmax".
            epsilon: Exploration probability for epsilon-greedy.
            epsilon_decay: Multiplicative decay per episode (1.0 = no decay).
            min_epsilon: Lower bound on epsilon.
            temperature: Softmax temperature (higher = more uniform).
            temperature_decay: Multiplicative decay per episode (1.0 = no decay).
            min_temperature: Lower bound on temperature.
        """
        if exploration not in EXPLORATION_STRATEGIES:
            raise ValueError(
                f"exploration must be one of {EXPLORATION_STRATEGIES}, got {exploration!r}"
            )
        self.n = n
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.q_table = np.zeros((num_states(n), num_actions))

    # ------------------------------------------------------------------
    # State indexing and action selection
    # ------------------------------------------------------------------

    def _get_state_index(self, observation: tuple) -> int:
        """Convert observation tuple to flat state index."""
        return state_to_index(observation, self.n)

    def _softmax_probs(self, q_values: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over Q-values."""
        shifted = q_values - np.max(q_values)
        exp_q = np.exp(shifted / self.temperature)
        return exp_q / exp_q.sum()

    def sample_action(self, observation: tuple, training: bool = True) -> int:
        """Select an action using the configured exploration strategy.

        Args:
            observation: (agent_x, agent_y, token_x, token_y, collected).
            training: If True, apply exploration; else act greedily.

        Returns:
            Action index in {0, 1, 2, 3}.
        """
        state_idx = self._get_state_index(observation)
        q_values = self.q_table[state_idx]

        if not training:
            return int(np.argmax(q_values))

        if self.exploration == "epsilon_greedy":
            if np.random.random() < self.epsilon:
                return np.random.randint(self.num_actions)
            return int(np.argmax(q_values))

        # softmax
        probs = self._softmax_probs(q_values)
        return int(np.random.choice(self.num_actions, p=probs))

    # ------------------------------------------------------------------
    # Decay helpers
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Apply multiplicative epsilon decay (call after each episode)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def decay_temperature(self) -> None:
        """Apply multiplicative temperature decay (call after each episode)."""
        self.temperature = max(
            self.min_temperature, self.temperature * self.temperature_decay
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save Q-table to a .npy file."""
        np.save(path, self.q_table)

    def load(self, path: str | Path) -> None:
        """Load Q-table from a .npy file."""
        self.q_table = np.load(path)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _exploration_str(self) -> str:
        if self.exploration == "softmax":
            return f"Temperature: {self.temperature:.4f}"
        return f"Epsilon: {self.epsilon:.4f}"

    def _log_progress(
        self,
        ep: int,
        num_episodes: int,
        rewards: list[float],
        successes: list[int],
        lengths: list[int],
        log_interval: int,
    ) -> None:
        recent_rewards = rewards[-log_interval:]
        recent_successes = successes[-log_interval:]
        recent_lengths = lengths[-log_interval:]
        print(
            f"Episode {ep}/{num_episodes} | "
            f"Avg reward: {np.mean(recent_rewards):.2f} | "
            f"Avg success rate: {np.mean(recent_successes):.1%} | "
            f"Avg episode length: {np.mean(recent_lengths):.1f} | "
            f"{self._exploration_str()}"
        )

    # ------------------------------------------------------------------
    # Evaluation (algorithm-independent)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int | None = None,
        seed: int = 0,
    ) -> dict:
        """Evaluate the agent greedily.

        Args:
            env: GridWorldEnv instance.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Optional step cap (falls back to env.max_steps).
            seed: Random seed used for the first episode reset.

        Returns:
            Dict with aggregate statistics and per-episode lists.
        """
        success_count = 0
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        success_lengths: list[int] = []
        max_steps = max_steps_per_episode or env.max_steps

        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed if i == 0 else None)
            total_reward = 0.0
            steps = 0
            while steps < max_steps:
                action = self.sample_action(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break

            success = info["reached_goal_with_token"]
            if success:
                success_count += 1
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            if success:
                success_lengths.append(steps)

        avg_success_length = (
            float(np.mean(success_lengths)) if success_lengths else float("nan")
        )
        std_success_length = (
            float(np.std(success_lengths)) if success_lengths else float("nan")
        )

        return {
            "avg_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "success_rate": success_count / num_episodes,
            "avg_episode_length": float(np.mean(episode_lengths)),
            "std_episode_length": float(np.std(episode_lengths)),
            "avg_success_episode_length": avg_success_length,
            "std_success_episode_length": std_success_length,
            "episode_rewards": list(episode_rewards),
            "episode_lengths": list(episode_lengths),
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seed": seed,
        }
