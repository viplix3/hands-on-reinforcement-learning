import logging
import numpy as np
from dataclasses import dataclass

from .base import ActionSelectionAlgo

logger = logging.getLogger(__name__)


@dataclass
class GradientBandit(ActionSelectionAlgo):
    """The Gradient Bandit action selection algorithm.
        This algortihm assigns a preference to each action and selects
        the action using a softmax function.

        The preference of an action has nothing to do with the actual
        reward of the action, i.e. if 100 is added to the preference
        of an action, the action would still generate the same reward
        (if selected) as it did before the preference was updated.

        Preferences are updated based on the reward of the action using
        a gradient ascent method.

    Args:
        num_arms (int): The number of arms in the bandit
        alpha (float): The learning rate
    """

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.reset()

    def select_action(self):
        """Selects an action having the highest softmax probability
        and updates the selected action's N value
        """
        softmax = np.exp(self.H) / np.sum(np.exp(self.H))
        selected_action_idx, selection_prob = self._sample_action(softmax)

    def update(self, action, reward):
        raise NotImplementedError

    def reset(self):
        """Resets the H values to 0"""
        # H = preference of each action
        self.H = np.zeros(self.num_arms)

    def _sample_action(self, softmax: np.ndarray):
        """Samples an action based on the softmax probabilities

        Args:
            softmax (np.ndarray): The softmax probabilities of each action
        Returns:
            selected_action_idx (int): The index of the selected action
            selection_prob (float): The probability of the selected action
        """
        selected_action_idx = np.argmax(softmax)
        selection_prob = softmax[selected_action_idx]
        logger.debug(f"Selected action {selected_action_idx}")
        return selected_action_idx, selection_prob
