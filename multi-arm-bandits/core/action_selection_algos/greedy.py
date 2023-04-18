import logging
import numpy as np

from .base import ActionSelectionAlgo

logger = logging.getLogger(__name__)


class Greedy(ActionSelectionAlgo):
    """The Greedy action selection algorithm.
        This algorithm always selects the action with best reward currently

    Args:
        num_arms (int): The number of arms in the bandit
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_action(self):
        """Selects the action with the highest estimated value
        and updates the selected action's N value
        """
        selected_action_idx = np.argmax(self.Q)
        logger.debug(f"Selected action {selected_action_idx}")
        self.N[selected_action_idx] += 1
        return selected_action_idx

    def update(self, action, reward):
        """Updates the selected action's R value and Q value"""
        self.rewards[action] += reward
        self.Q[action] = self.rewards[action] / self.N[action]
