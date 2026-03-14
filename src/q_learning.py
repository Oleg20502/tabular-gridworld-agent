"""Tabular Q-learning agent for GridWorld."""

from pathlib import Path

import numpy as np

from .state_utils import num_states, state_to_index


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        n: int,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
    ):
        """Initialize the Q-learning agent.

        Args:
            n: Grid size (NxN). Used to compute num_states and for state_to_index.
            num_actions: Number of actions (default 4).
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Exploration probability (epsilon-greedy).
            epsilon_decay: Multiplicative decay per episode (1.0 = no decay).
            min_epsilon: Minimum exploration rate.
        """
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        n_states = num_states(n)
        self.q_table = np.zeros((n_states, num_actions))

    def _get_state_index(self, observation: tuple) -> int:
        """Convert observation tuple to state index."""
        return state_to_index(observation, self.n)

    def sample_action(self, observation: tuple, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            observation: (agent_x, agent_y, token_x, token_y, collected).
            training: If True, use epsilon-greedy; else use greedy.

        Returns:
            Action index in {0, 1, 2, 3} (up, down, right, left).
        """
        state_idx = self._get_state_index(observation)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(4)
        return int(np.argmax(self.q_table[state_idx]))

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

        episode_rewards: list[float] = []
        success_count = 0
        max_steps = max_steps_per_episode or self.n**2 * 4

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

            episode_rewards.append(total_reward)
            if terminated:
                success_count += 1
            self.decay_epsilon()

            if (ep + 1) % log_interval == 0:
                recent = episode_rewards[-log_interval:]
                print(
                    f"Episode {ep + 1}/{num_episodes} | "
                    f"Avg reward: {np.mean(recent):.2f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        return episode_rewards, success_count


    def evaluate(self, env, num_episodes: int, max_steps_per_episode: int | None = None, seed: int | None = None) -> float:
        """Evaluate the agent."""
        success_count = 0
        episode_rewards: list[float] = []
        max_steps = max_steps_per_episode or self.n**2 * 4
        for i in range(num_episodes):
            obs, _ = env.reset(seed=None if seed is None else seed + i)
            total_reward = 0.0
            steps = 0
            while steps < max_steps:
                action = self.sample_action(obs, training=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            
            if terminated:
                success_count += 1
            
            episode_rewards.append(total_reward)
        success_rate = success_count / num_episodes
        avg_reward = float(np.mean(episode_rewards))

        results = {
            "avg_reward": avg_reward,
            "std_reward": float(np.std(episode_rewards)),
            "success_rate": success_rate,
            "episode_rewards": list(episode_rewards),
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seed": seed,
        }

        return results