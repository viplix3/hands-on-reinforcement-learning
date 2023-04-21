import numpy as np


def generate_rewards(num_actions):
    """Generates rewards for each action using a Gaussian distribution

    Args:
        num_actions (int): Number of actions
    Returns:
        actual_rewards (list): Rewards for each action
    """
    # Generate the actual rewards for each arm
    min_mean, max_mean = -1, 1
    action_reward_means = (max_mean - min_mean) * np.random.random(
        num_actions
    ) + min_mean
    actual_rewards = []

    # Generate rewards for each arm using a Gaussian distribution with the
    # mean as the point and a std of 1
    for mean in action_reward_means:
        reward = np.random.normal(mean, 1)
        actual_rewards.append(reward)

    return actual_rewards
