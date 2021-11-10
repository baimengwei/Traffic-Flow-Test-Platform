from abc import ABCMeta, abstractmethod
from configs.conf_path import ConfPath


class AgentFix(metaclass=ABCMeta):
    """
        An abstract class for value based method
    """

    def __init__(self, conf_path: ConfPath, round_number: int, inter_name: str):
        """
        """
        pass

    @staticmethod
    def convert_state_to_input(conf_traffic, state):
        pass


    @abstractmethod
    def choose_action(self, state,choice_random=False):
        pass
