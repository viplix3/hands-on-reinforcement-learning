from enum import Enum


class ActionSelectionAlgoTypes(Enum):
    GREEDY = 0
    E_GREEDY = 1
    GRADIENT_BANDIT = 2
    UCB = 3


class ActionSelectionAlgoFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_action_selection_algo(algo_type, **kwargs):
        if algo_type == ActionSelectionAlgoTypes.GREEDY:
            from .greedy import Greedy

            return Greedy(**kwargs)
        elif algo_type == ActionSelectionAlgoTypes.E_GREEDY:
            from .e_greedy import EpsilonGreedy

            return EpsilonGreedy(**kwargs)
        elif algo_type == ActionSelectionAlgoTypes.GRADIENT_BANDIT:
            from .gradient_bandit import GradientBandit

            return GradientBandit(**kwargs)
        elif algo_type == ActionSelectionAlgoTypes.UCB:
            from .ucb import UpperConfidenceBound

            return UpperConfidenceBound(**kwargs)
        else:
            raise ValueError("Invalid action selection algorithm type")
