import numpy as np


def generate_rewards_without_variance(num_actions):
    """Generates rewards for each action using a Gaussian distribution
        Rewards once generated are fixed and do not change for each step

    Args:
        num_actions (int): Number of actions
    Returns:
        actual_rewards (list): Rewards for each action
    """
    # Generate the actual rewards for each arm
    min_mean, max_mean = -1.5, 1.5
    action_reward_means = (max_mean - min_mean) * np.random.random(
        num_actions
    ) + min_mean
    return action_reward_means


def generate_rewards_with_variance(num_actions, num_steps):
    """Generates rewards for each action using a Gaussian distribution
        Rewards are generated for each step with a standard deviation of 5

    Args:
        num_actions (int): Number of actions
        num_steps (int): Number of steps
    Returns:
        actual_rewards (list): Rewards for each action
    """
    min_mean, max_mean = -1.5, 1.5
    action_reward_means = (max_mean - min_mean) * np.random.random(
        num_actions
    ) + min_mean

    actual_rewards = []
    for mean in action_reward_means:
        reward = np.random.normal(mean, 5, size=num_steps)
        actual_rewards.append(reward)
    return actual_rewards
