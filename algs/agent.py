import numpy as np
from abc import ABCMeta, abstractmethod
from configs.conf_path import ConfPath


class Agent(metaclass=ABCMeta):
    """
        An abstract class for value based method
    """

    def __init__(self, conf_path: ConfPath, round_number: int, inter_name: str):
        """
        """
        pass

    @staticmethod
    def convert_state_to_input(conf_traffic, state):
        input = []
        list_feature = conf_traffic.FEATURE
        dic_phase_expansion = conf_traffic.TRAFFIC_INFO['phase_lane_mapping']
        for feature in list_feature:
            if feature == "cur_phase":
                input.append(np.array(dic_phase_expansion[state[feature] - 1]))
            else:
                input.append(np.array(state[feature]))
        return input

    @staticmethod
    def decay_epsilon(conf_agent, round_number):
        """For value based method.
        When reached round i = (log(eps_min)-log(eps_init)) / log(decay)
        eps_init reached eps_min. default round: 27
        """
        decayed_epsilon = conf_agent['EPSILON'] * \
                          np.power(conf_agent['EPSILON_DECAY'],
                                   round_number)
        conf_agent.EPSILON = max(decayed_epsilon,
                                 conf_agent['MIN_EPSILON'])
        return conf_agent

    @abstractmethod
    def choose_action(self, state, choice_random: bool):
        pass

    @abstractmethod
    def build_network(self):
        pass

    @abstractmethod
    def build_network_bar(self):
        pass

    @abstractmethod
    def load_network(self, round_number):
        pass

    @abstractmethod
    def load_network_bar(self, round_number):
        pass

    @abstractmethod
    def prepare_Xs_Y(self, sample_set):
        pass

    @abstractmethod
    def train_network(self):
        pass

    @abstractmethod
    def save_network(self, round_number):
        pass
