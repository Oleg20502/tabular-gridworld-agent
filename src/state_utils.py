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


def index_to_state(idx: int, n: int) -> Tuple[int, int, int, int, int]:
    """
    Map state index back to observation tuple.
    Returns (agent_x, agent_y, token_x, token_y, collected).
    For collected=1, token coords are 0 (dummy).
    """
    num_not_collected = n**4
    if idx < num_not_collected:
        remainder = idx
        ty = remainder % n
        remainder //= n
        tx = remainder % n
        remainder //= n
        ay = remainder % n
        remainder //= n
        ax = remainder
        return (ax, ay, tx, ty, 0)
    else:
        idx_collected = idx - num_not_collected
        ay = idx_collected % n
        ax = idx_collected // n
        return (ax, ay, 0, 0, 1)
