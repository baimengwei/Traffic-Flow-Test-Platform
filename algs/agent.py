import numpy as np
from abc import ABCMeta, abstractmethod
from configs.conf_path import ConfPath


class Agent(classmethod=ABCMeta):
    """
        An abstract class for value based method
    """

    def __init__(self, conf_path: ConfPath, round_number: int):
        """
        """
        _, self.conf_agent, self.conf_traffic = conf_path.load_conf_file()
        self.conf_path = conf_path
        self.round_number = round_number
        self.__decay_epsilon()

    def convert_state_to_input(self, state, algorithm):
        input = []
        list_feature = self.conf_traffic.FEATURE
        dic_phase_expansion = self.conf_traffic.LANE_PHASE_INFO['phase_lane_mapping']
        for feature in list_feature:
            if feature == "cur_phase":
                input.append(np.array(dic_phase_expansion[state[feature] - 1]))
            else:
                input.append(np.array(state[feature]))
        return input

    def __decay_epsilon(self):
        """For value based method.
        When reached round i = (log(eps_min)-log(eps_init)) / log(decay)
        eps_init reached eps_min. default round: 27
        """
        decayed_epsilon = self.conf_agent.EPSILON * \
                          np.power(self.conf_agent.EPSILON_DECAY,
                                   self.round_number)
        self.conf_agent.EPSILON = max(decayed_epsilon,
                                      self.conf_agent.MIN_EPSILON)

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
    def load_network(self, file_name):
        pass

    @abstractmethod
    def load_network_bar(self, file_name):
        pass

    @abstractmethod
    def prepare_Xs_Y(self, sample_set):
        pass

    @abstractmethod
    def train_network(self):
        pass

    @abstractmethod
    def save_network(self, file_name):
        pass
