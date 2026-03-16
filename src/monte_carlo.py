"""Tabular Monte Carlo control agent for GridWorld."""

import numpy as np

from ._base import BaseTabularAgent


class MonteCarloAgent(BaseTabularAgent):
    """On-policy first-visit Monte Carlo control.

    Collects a complete episode under the current epsilon-greedy or softmax
    policy, then updates every first-visited (s, a) pair using the
    discounted return G from that time-step onwards:

        Q(s, a) <- Q(s, a) + alpha * (G - Q(s, a))

    Returns are computed efficiently in a single backward pass over the
    trajectory, so no extra storage of per-step returns is required.
    """

    def update_from_episode(
        self, trajectory: list[tuple[tuple, int, float]]
    ) -> None:
        """Update Q-table from a complete episode.

        Args:
            trajectory: Ordered list of (observation, action, reward) tuples
                collected during the episode.
        """
        G = 0.0
        visited: set[tuple[int, int]] = set()

        for obs, action, reward in reversed(trajectory):
            G = reward + self.gamma * G
            s = self._get_state_index(obs)
            if (s, action) not in visited:
                visited.add((s, action))
                self.q_table[s, action] += self.alpha * (G - self.q_table[s, action])

    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int | None = None,
        log_interval: int = 100,
        seed: int | None = None,
    ) -> dict:
        """Train the agent using first-visit MC control.

        Args:
            env: GridWorldEnv instance.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Optional step limit (falls back to env.max_steps).
            log_interval: Print progress every N episodes.
            seed: Random seed for the first episode reset.

        Returns:
            Dict with ``episode_rewards``, ``success_counts``, ``episode_lengths``.
        """
        if seed is not None:
            np.random.seed(seed)

        episode_rewards: list[float] = []
        success_counts: list[int] = []
        episode_lengths: list[int] = []
        max_steps = max_steps_per_episode or env.max_steps

        for ep in range(num_episodes):
            obs, info = env.reset(seed=seed if ep == 0 else None)
            trajectory: list[tuple[tuple, int, float]] = []
            total_reward = 0.0
            steps = 0

            # --- collect a full episode -----------------------------------
            while steps < max_steps:
                action = self.sample_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append((obs, action, reward))
                total_reward += reward
                obs = next_obs
                steps += 1
                if terminated or truncated:
                    break

            # --- batch Q update from the completed trajectory -------------
            self.update_from_episode(trajectory)

            episode_rewards.append(total_reward)
            success_counts.append(1 if info["reached_goal_with_token"] else 0)
            episode_lengths.append(steps)

            self.decay_epsilon()
            self.decay_temperature()

            if (ep + 1) % log_interval == 0:
                self._log_progress(
                    ep + 1, num_episodes,
                    episode_rewards, success_counts, episode_lengths,
                    log_interval,
                )

        return {
            "episode_rewards": list(episode_rewards),
            "success_counts": list(success_counts),
            "episode_lengths": list(episode_lengths),
        }
