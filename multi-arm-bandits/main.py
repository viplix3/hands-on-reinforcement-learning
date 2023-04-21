import os
import logging
import argparse
import numpy as np
from tqdm import tqdm


from core.action_selection_algos.factory import (
    ActionSelectionAlgoTypes,
    ActionSelectionAlgoFactory,
)
from utils.viz_utils import plot_rewards, plot_average_rewards
from utils.misc import generate_rewards_with_variance, generate_rewards_without_variance

logger = logging.getLogger(__name__)


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="debug")
    parser.add_argument("--plots_dir", type=str, default="plots")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.2)
    return parser.parse_args()


def multiarm_bandit(args: argparse.Namespace):
    logger.info("Starting Multi-Arm Bandit")
    logger.info(f"Arguments: {args}")

    rewards_without_variance = generate_rewards_without_variance(args.num_arms)
    rewards_with_variance = generate_rewards_with_variance(
        args.num_arms, args.num_steps
    )

    plot_rewards(
        args.num_arms,
        rewards_without_variance,
        "rewards_without_variance",
        os.path.join(os.getcwd(), args.plots_dir),
    )
    plot_rewards(
        args.num_arms,
        rewards_with_variance,
        "rewards_with_variance",
        os.path.join(os.getcwd(), args.plots_dir),
    )

    algo_factory = ActionSelectionAlgoFactory()
    algorithms = []

    for algo_type in ActionSelectionAlgoTypes:
        action_selection_algo = algo_factory.get_action_selection_algo(
            algo_type, **vars(args)
        )
        algorithms.append(action_selection_algo)

    # Run algorithms using rewards without variance
    for algo in algorithms:
        algo.reset()

    for _ in tqdm(range(args.num_steps)):
        for algo in algorithms:
            action = algo.select_action()

            reward = rewards_without_variance[action]
            algo.update(action, reward)

    plot_average_rewards(
        algorithms,
        os.path.join(os.getcwd(), args.plots_dir),
        fig_name="average_rewards_without_variance",
    )

    # Run algorithms with non stationary rewards
    for algo in algorithms:
        algo.reset()

    for step in tqdm(range(args.num_steps)):
        for algo in algorithms:
            action = algo.select_action()

            reward = rewards_with_variance[action][step]
            algo.update(action, reward)

    plot_average_rewards(
        algorithms,
        os.path.join(os.getcwd(), args.plots_dir),
        fig_name="average_rewards_with_variance",
    )


if __name__ == "__main__":
    args = parse_cmd_args()
    logdir = os.path.join(os.path.dirname(__file__), "logs")
    logfile = os.path.join(logdir, "app.log")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    logging.basicConfig(
        filename=logfile,
        filemode="w",
        level=args.log_level.upper()
        if isinstance(args.log_level, str)
        else args.log_level,
        format="%(asctime)s [%(levelname)8s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # To make the results reproducible
    np.random.seed(2000)
    multiarm_bandit(args)
