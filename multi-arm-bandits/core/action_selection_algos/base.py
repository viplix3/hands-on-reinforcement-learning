import numpy as np
from abc import ABC, abstractmethod


class ActionSelectionAlgo(ABC):
    """Abstract class for action selection algorithms."""

    def __init__(self, num_arms: int = 10, **kwargs):
        self.num_arms = num_arms
        self.reset()

    @abstractmethod
    def select_action(self):
        """Select an action to play.

        Returns:
            int: The action to play.
        """
        pass

    @abstractmethod
    def update(self, action, reward):
        """Update the action selection algorithm.

        Args:
            action (int): The action played.
            reward (float): The reward obtained.
        """
        pass

    def reset(self):
        """Resets the Q, N, and R values to 0"""
        # Q(a) = estimated value of action a
        # N(a) = number of times action a was selected
        # R(a) = sum of rewards of action a
        self.Q = np.zeros(self.num_arms)
        self.N = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        self.average_rewards = []

    def update_average_reward(self):
        """Updates the average reward of the bandit."""
        self.average_rewards.append(np.sum(self.rewards) / np.sum(self.N))
