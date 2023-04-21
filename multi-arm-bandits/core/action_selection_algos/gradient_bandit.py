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

        self.algo_name = "Gradient Bandit"
        self.alpha = alpha
        self.reset()

    def select_action(self):
        """Selects an action having the highest softmax probability
        and updates the selected action's N value
        """
        softmax = np.exp(self.H) / np.sum(np.exp(self.H))
        selected_action_idx, self.selection_prob = self._sample_action(softmax)
        logger.debug(f"Selected action {selected_action_idx}")
        self.N[selected_action_idx] += 1
        return selected_action_idx

    def update(self, action, reward):
        # Gradient ascent method to update the preferences
        baseline_reward = (
            self.average_rewards[-1] if len(self.average_rewards) > 0 else 0
        )
        selected_action_mask = np.zeros(self.num_arms, dtype=int)
        selected_action_mask[action] = 1

        self.H[selected_action_mask == 1] += (
            self.alpha * (reward - baseline_reward) * (1 - self.selection_prob)
        )

        self.H[selected_action_mask == 0] -= (
            self.alpha * (reward - baseline_reward) * self.selection_prob
        )

        self.rewards[action] += reward
        self.N[action] += 1
        self.Q[action] = self.rewards[action] / self.N[action]
        self.update_average_reward()

    def reset(self):
        """Resets the H values to 0"""
        super().reset()
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
        # Why not simply use np.argmax(softmax) ?
        # Using np.random.choice with the softmax probabilities allows the
        # algorithm to explore different actions with a probability
        # proportional to their softmax values. This way, the algorithm
        # balances exploration # and exploitation, as actions with higher
        # preferences are more likely to be selected, but there's still
        # a chance to try other actions as well.

        # By always selecting the action with the highest probability, the
        # algorithm may not sufficiently explore other actions and may fail
        # to find the best possible action in certain cases. This can lead
        # to suboptimal performance.
        selected_action_idx = np.random.choice(np.arange(len(softmax)), p=softmax)
        selection_prob = softmax[selected_action_idx]
        return selected_action_idx, selection_prob
