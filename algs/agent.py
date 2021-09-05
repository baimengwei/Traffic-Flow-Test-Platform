import numpy as np


class Agent:
    """
        An abstract class for FRAPPlus
    """

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 round_number):
        """
        Args:
            dic_agent_conf:
            dic_traffic_env_conf: should be item rather than list
            dic_path: should be item rather than list
        Returns:
            saved value
        """
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

    def decay_epsilon(self, round_number):
        """For value based method.

        Warning: MODIFIED DIC_AGENT_CONF : EPSILON VALUE
        When reached round i = (log(eps_min)-log(eps_init)) / log(decay)
        eps_init reached eps_min. default round: 27
        """
        decayed_epsilon = \
            self.dic_agent_conf["EPSILON"] * \
            np.power(self.dic_agent_conf["EPSILON_DECAY"], round_number)
        self.dic_agent_conf["EPSILON"] = max(
            decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def decay_noise(self, round_number):
        """For TD3
        Warning: MODIFIED DIC_AGENT_CONF : NOISE VALUE
        """
        decayed_expl_noise = \
            self.dic_agent_conf["EXPL_NOISE"] * \
            np.power(self.dic_agent_conf["EXPL_NOISE_DECAY"], round_number)
        self.dic_agent_conf["EXPL_NOISE"] = max(
            decayed_expl_noise, self.dic_agent_conf["EXPL_NOISE_END"])

    def choose_action(self, state, choice_random):
        raise NotImplementedError

    def build_network(self):
        raise NotImplementedError

    def build_network_bar(self):
        raise NotImplementedError

    def load_network(self, file_name):
        raise NotImplementedError

    def load_network_bar(self, file_name):
        raise NotImplementedError

    def prepare_Xs_Y(self, sample_set):
        raise NotImplementedError

    def train_network(self):
        raise NotImplementedError

    def save_network(self, file_name):
        raise NotImplementedError
