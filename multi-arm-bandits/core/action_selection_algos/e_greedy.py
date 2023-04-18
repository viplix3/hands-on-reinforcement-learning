import logging
import numpy as np
from dataclasses import dataclass

from .base import ActionSelectionAlgo

logger = logging.getLogger(__name__)


@dataclass
class EpsilonGreedy(ActionSelectionAlgo):
    """The Epsilon Greedy action selection algorithm.
        This algorithm selects the action with best reward currently with
        probability 1 - epsilon and selects a random action with probability
        epsilon

        This would work better than the greedy algorithm if the rewards are
        not deterministic

    Args:
        num_arms (int): The number of arms in the bandit
        epsilon (float): The probability of selecting a random action
    """

    def __init__(self, epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon

    def select_action(self):
        """Selects the action with the highest estimated value with
        probability 1 - epsilon and selects a random action with
        probability epsilon and updates the selected action's N value
        """
        if np.random.random() < self.epsilon:
            selected_action_idx = np.random.randint(self.num_arms)
        else:
            selected_action_idx = np.argmax(self.Q)
        logger.debug(f"Selected action {selected_action_idx}")
        self.N[selected_action_idx] += 1
        return selected_action_idx

    def update(self, action, reward):
        """Updates the selected action's R value and Q value"""
        self.rewards[action] += reward
        self.Q[action] = self.rewards[action] / self.N[action]
