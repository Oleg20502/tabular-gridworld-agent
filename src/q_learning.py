"""Tabular Q-learning agent for GridWorld."""

from pathlib import Path

import numpy as np

from .state_utils import num_states, state_to_index


EXPLORATION_STRATEGIES = ("epsilon_greedy", "softmax")


class QLearningAgent:
    """Tabular Q-learning agent with pluggable exploration strategies."""

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
        """Initialize the Q-learning agent.

        Args:
            n: Grid size (NxN). Used to compute num_states and for state_to_index.
            num_actions: Number of actions (default 4).
            alpha: Learning rate.
            gamma: Discount factor.
            exploration: Exploration strategy — "epsilon_greedy" or "softmax".
            epsilon: Exploration probability for epsilon-greedy.
            epsilon_decay: Multiplicative decay per episode (1.0 = no decay).
            min_epsilon: Minimum epsilon.
            temperature: Softmax temperature (higher = more uniform).
            temperature_decay: Multiplicative decay per episode (1.0 = no decay).
            min_temperature: Minimum temperature.
        """
        if exploration not in EXPLORATION_STRATEGIES:
            raise ValueError(f"exploration must be one of {EXPLORATION_STRATEGIES}, got {exploration!r}")
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        n_states = num_states(n)
        self.q_table = np.zeros((n_states, num_actions))

    def _get_state_index(self, observation: tuple) -> int:
        """Convert observation tuple to state index."""
        return state_to_index(observation, self.n)

    def _softmax_probs(self, q_values: np.ndarray) -> np.ndarray:
        """Compute numerically stable softmax probabilities over Q-values."""
        shifted = q_values - np.max(q_values)
        exp_q = np.exp(shifted / self.temperature)
        return exp_q / exp_q.sum()

    def sample_action(self, observation: tuple, training: bool = True) -> int:
        """Select action using the configured exploration strategy.

        Args:
            observation: (agent_x, agent_y, token_x, token_y, collected).
            training: If True, apply exploration; else use greedy.

        Returns:
            Action index in {0, 1, 2, 3} (up, down, right, left).
        """
        state_idx = self._get_state_index(observation)
        q_values = self.q_table[state_idx]

        if not training:
            return int(np.argmax(q_values))

        if self.exploration == "epsilon_greedy":
            if np.random.random() < self.epsilon:
                return np.random.randint(len(q_values))
            return int(np.argmax(q_values))

        # softmax
        probs = self._softmax_probs(q_values)
        return int(np.random.choice(len(q_values), p=probs))

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        next_obs: tuple,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Perform Q-learning update.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        """
        state_idx = self._get_state_index(obs)
        next_state_idx = self._get_state_index(next_obs)

        if terminated or truncated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        td_error = target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Apply epsilon decay (call after each episode)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def decay_temperature(self) -> None:
        """Apply temperature decay (call after each episode)."""
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)

    def save(self, path: str | Path) -> None:
        """Save Q-table to file."""
        np.save(path, self.q_table)

    def load(self, path: str | Path) -> None:
        """Load Q-table from file."""
        self.q_table = np.load(path)

    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int | None = None,
        log_interval: int = 100,
        seed: int | None = None,
    ) -> tuple[list[float], int]:
        """Train the agent.

        Args:
            env: GridWorldEnv instance.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Optional step limit (env may have its own).
            log_interval: Print progress every N episodes.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (episode rewards, success count).
        """
        if seed is not None:
            np.random.seed(seed)

        episode_rewards = []
        success_counts = []
        episode_lengths = []
        max_steps = max_steps_per_episode or env.max_steps

        for ep in range(num_episodes):
            obs, info = env.reset(seed=seed if ep == 0 else None)
            total_reward = 0.0
            steps = 0
            while steps < max_steps:
                action = self.sample_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)

                self.update(obs, action, reward, next_obs, terminated, truncated)
                
                total_reward += reward
                obs = next_obs
                steps += 1
                if terminated or truncated:
                    break
            
            episode_lengths.append(steps)
            episode_rewards.append(total_reward)

            if info["reached_goal_with_token"]:
                success_counts.append(1)
            else:
                success_counts.append(0)
            
            self.decay_epsilon()
            self.decay_temperature()

            if (ep + 1) % log_interval == 0:
                recent_rewards = episode_rewards[-log_interval:]
                recent_success_counts = success_counts[-log_interval:]
                recent_episode_lengths = episode_lengths[-log_interval:]
                if self.exploration == "softmax":
                    explore_str = f"Temperature: {self.temperature:.4f}"
                else:
                    explore_str = f"Epsilon: {self.epsilon:.4f}"
                print(
                    f"Episode {ep + 1}/{num_episodes} | "
                    f"Avg reward: {np.mean(recent_rewards):.2f} | "
                    f"Avg success rate: {np.mean(recent_success_counts) / len(recent_success_counts) * 100:.1%}% | "
                    f"Avg episode length: {np.mean(recent_episode_lengths):.1f} | "
                    f"{explore_str}"
                )

        logs = {
            "episode_rewards": list(episode_rewards),
            "success_counts": list(success_counts),
            "episode_lengths": list(episode_lengths),
        }
        return logs


    def evaluate(self, env, num_episodes: int, max_steps_per_episode: int | None = None, seed: int = 0) -> float:
        """Evaluate the agent."""
        success_count = 0
        episode_rewards = []
        episode_lengths = []
        success_lengths = []
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

        success_rate = success_count / num_episodes
        avg_reward = float(np.mean(episode_rewards))
        avg_episode_length = float(np.mean(episode_lengths))
        std_episode_length = float(np.std(episode_lengths))

        avg_success_length = float(np.mean(success_lengths)) if success_lengths else float("nan")
        std_success_length = float(np.std(success_lengths)) if success_lengths else float("nan")

        results = {
            "avg_reward": avg_reward,
            "std_reward": float(np.std(episode_rewards)),
            "success_rate": success_rate,
            "avg_episode_length": avg_episode_length,
            "std_episode_length": std_episode_length,
            "avg_success_episode_length": avg_success_length,
            "std_success_episode_length": std_success_length,
            "episode_rewards": list(episode_rewards),
            "episode_lengths": list(episode_lengths),
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seed": seed,
        }

        return results