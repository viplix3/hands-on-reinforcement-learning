import logging
import numpy as np
from dataclasses import dataclass

from .base import ActionSelectionAlgo

logger = logging.getLogger(__name__)


@dataclass
class UpperConfidenceBound(ActionSelectionAlgo):
    """The Upper Confidence Bound action selection algorithm.
        The epsilon greedy algorithm selects action for exploration
        without any preferance.

        The UCB algorithm selects the action with highest most uncertainity
        in its estimated value.

    Args:
        num_arms (int): The number of arms in the bandit
        c (float): The exploration parameter
    """

    def __init__(self, c: float = 0.2, **kwargs):
        super().__init__(**kwargs)

        self.algo_name = "Upper Confidence Bound"
        self.c = c
        self.reset()

    def select_action(self):
        """Selects the action having the highest upper confidence bound
        and updates the selected action's N value

        UCB = Q + c * sqrt(ln(t) / N)
        """
        ucb = self.Q + self.c * np.sqrt(np.log(self.t + 1) / (self.N + 1e-6))
        selected_action_idx = np.random.choice(np.flatnonzero(ucb == ucb.max()))
        logger.debug(f"Selected action {selected_action_idx}")
        self.N[selected_action_idx] += 1
        self.t += 1
        return selected_action_idx

    def update(self, action, reward):
        """Updates the selected action's R value and Q value"""
        self.rewards[action] += reward
        self.Q[action] = self.rewards[action] / self.N[action]
        self.update_average_reward()

    def reset(self):
        """Resets the Q, N, and R values to 0"""
        super().reset()
        self.t = 0  # t = number of times select_action() was called
