"""Tabular SARSA agent for GridWorld."""

import numpy as np

from ._base import BaseTabularAgent


class SARSAAgent(BaseTabularAgent):
    """On-policy TD control (SARSA).

    Update rule:
        Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))

    where a' is the action *actually selected* by the policy in s', making
    this on-policy (unlike Q-learning which bootstraps from max_a' Q(s', a')).
    """

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        next_obs: tuple,
        next_action: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Perform a single SARSA update.

        Args:
            obs: Current observation.
            action: Action taken in obs.
            reward: Reward received.
            next_obs: Resulting observation.
            next_action: Action already selected from next_obs by the policy.
            terminated: Whether the episode ended due to a terminal state.
            truncated: Whether the episode was cut short by a step limit.
        """
        s = self._get_state_index(obs)
        s_next = self._get_state_index(next_obs)

        if terminated or truncated:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[s_next, next_action]

        self.q_table[s, action] += self.alpha * (target - self.q_table[s, action])

    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int | None = None,
        log_interval: int = 100,
        seed: int | None = None,
    ) -> dict:
        """Train the agent using SARSA.

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
            # SARSA: select the first action before the loop so both (s, a)
            # and (s', a') are available at update time.
            action = self.sample_action(obs, training=True)
            total_reward = 0.0
            steps = 0

            while steps < max_steps:
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_action = self.sample_action(next_obs, training=True)

                self.update(obs, action, reward, next_obs, next_action, terminated, truncated)

                total_reward += reward
                obs = next_obs
                action = next_action
                steps += 1
                if terminated or truncated:
                    break

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
