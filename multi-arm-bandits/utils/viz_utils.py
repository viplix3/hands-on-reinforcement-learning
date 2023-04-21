import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


def plot_rewards(
    num_actions: int, rewards: Union[float, List[float]], fig_name: str, save_path: str
):
    """Visualizes actions and their corresponding rewards and saves the figure

    Args:
        num_actions (int): Number of actions
        rewards (Union[float, List[float]]): Rewards for each action
        fig_name (str): Name of the figure
        save_path (str): Path to save the figure
    """
    # Figure with actions on x-axis and rewards distribution on y-axis
    fig, ax = plt.subplots()
    ax.set_xlabel("Actions")
    ax.set_ylabel("Rewards")
    ax.set_title("Rewards for each action")
    ax.set_xticks(range(num_actions))
    ax.set_xticklabels(range(num_actions))
    # ax.set_ylim(-4, 4)

    for i in range(num_actions):
        ax.axvline(i, color="k", linestyle="--", alpha=0.3)

        if isinstance(rewards[i], np.ndarray):
            ax.scatter([i] * len(rewards[i]), rewards[i], marker="o", s=10)
        else:
            ax.scatter(i, rewards[i], marker="o", s=10)

    fig.savefig(f"{save_path}/{fig_name}.png")


def plot_average_rewards(algorithms: List, save_path: str, fig_name: str):
    """Plots the average reward for each step and saves the figure

    Args:
        algorithms (List): List of action selection algorithms
        save_path (str): Path to save the figure
        fig_name (str): Name of the figure
    """
    # Figure with steps on x-axis and average reward on y-axis
    fig, ax = plt.subplots()
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward vs Steps")
    for algo in algorithms:
        ax.plot(algo.average_rewards, label=algo.algo_name)
    ax.legend()
    fig.savefig(f"{save_path}/{fig_name}.png")
