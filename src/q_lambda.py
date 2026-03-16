"""Watkins's Q(λ) agent for GridWorld."""

import numpy as np

from ._base import BaseTabularAgent


class QLambdaAgent(BaseTabularAgent):
    """Watkins's Q(λ): off-policy TD control with eligibility traces.

    Combines the off-policy Q-learning backup (bootstrapping from the
    greedy action in the next state) with eligibility traces for faster
    credit assignment.  The key distinction from SARSA(λ) is that traces
    are **cut to zero** whenever the agent takes a non-greedy (exploratory)
    action, keeping the algorithm off-policy and convergent:

        delta_t = r_t + gamma * max_a' Q(s', a') - Q(s, a)
        e(s, a) += 1                           # accumulating trace

        Q(s, a) += alpha * delta_t * e(s, a)   # for all (s, a)

        if greedy action was taken:
            e(s, a) *= gamma * lambda          # decay traces
        else:
            e(s, a)  = 0                       # Watkins's cut

    Reference: Watkins (1989) "Learning from Delayed Rewards", Ch. 6.
    """

    def __init__(self, *args, lam: float = 0.9, **kwargs):
        """
        Args:
            lam: Trace decay parameter λ ∈ [0, 1].
                 λ=0 reduces to one-step Q-learning; λ=1 gives full
                 Monte-Carlo-style returns (while remaining off-policy).
            *args / **kwargs: Forwarded to BaseTabularAgent.__init__.
        """
        super().__init__(*args, **kwargs)
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam = lam
        self.eligibility = np.zeros_like(self.q_table)

    def reset_traces(self) -> None:
        """Zero out all eligibility traces (called at the start of each episode)."""
        self.eligibility[:] = 0.0

    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int | None = None,
        log_interval: int = 100,
        seed: int | None = None,
    ) -> dict:
        """Train the agent using Watkins's Q(λ).

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
            self.reset_traces()
            obs, info = env.reset(seed=seed if ep == 0 else None)
            total_reward = 0.0
            steps = 0

            while steps < max_steps:
                action = self.sample_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)

                s = self._get_state_index(obs)
                s_next = self._get_state_index(next_obs)
                greedy_next = int(np.argmax(self.q_table[s_next]))

                if terminated or truncated:
                    delta = reward - self.q_table[s, action]
                else:
                    delta = (
                        reward
                        + self.gamma * self.q_table[s_next, greedy_next]
                        - self.q_table[s, action]
                    )

                # Accumulating eligibility trace for the visited (s, a).
                self.eligibility[s, action] += 1.0

                # Vectorised update over the entire Q-table.
                self.q_table += self.alpha * delta * self.eligibility

                if terminated or truncated:
                    self.reset_traces()
                elif action == greedy_next:
                    # Greedy action taken — decay traces normally.
                    self.eligibility *= self.gamma * self.lam
                else:
                    # Exploratory action taken — cut traces (Watkins's rule).
                    self.reset_traces()

                total_reward += reward
                obs = next_obs
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
