import os
import logging
import argparse
import numpy as np
from tqdm import tqdm


from core.action_selection_algos.factory import (
    ActionSelectionAlgoTypes,
    ActionSelectionAlgoFactory,
)

logger = logging.getLogger(__name__)


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="debug")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=2000)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1)
    return parser.parse_args()


def multiarm_bandit(args: argparse.Namespace):
    logger.info("Starting Multi-Arm Bandit")
    logger.info(f"Arguments: {args}")

    algo_factory = ActionSelectionAlgoFactory()
    algorithms = []

    for algo_type in ActionSelectionAlgoTypes:
        action_selection_algo = algo_factory.get_action_selection_algo(
            algo_type, **vars(args)
        )
        algorithms.append(action_selection_algo)

    for _ in tqdm(range(args.num_runs)):
        for algo in algorithms:
            action = algo.select_action()

        reward = np.random.random()
        for algo in algorithms:
            algo.update(action, reward)


if __name__ == "__main__":
    args = parse_cmd_args()
    logdir = os.path.join(os.path.dirname(__file__), "logs")
    logfile = os.path.join(logdir, "app.log")

    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        filemode="w",
        level=args.log_level.upper()
        if isinstance(args.log_level, str)
        else args.log_level,
        format="%(asctime)s [%(levelname)8s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    multiarm_bandit(args)
