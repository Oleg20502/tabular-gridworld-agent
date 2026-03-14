"""State indexing utilities for reduced state space representation."""

from typing import Tuple


def num_states(n: int) -> int:
    """Return the number of distinct states in the reduced state space."""
    return n**4 + n**2


def state_to_index(obs: Tuple[int, int, int, int, int], n: int) -> int:
    """
    Map observation tuple to unique state index.
    obs = (agent_x, agent_y, token_x, token_y, collected)
    """
    ax, ay, tx, ty, collected = obs
    if collected == 0:
        return ax * (n**3) + ay * (n**2) + tx * n + ty
    else:
        return n**4 + ax * n + ay

