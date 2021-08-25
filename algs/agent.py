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

        self.decay_epsilon(self.round_number)
        self.decay_lr(self.round_number)

        # self._decouple_params()
        # self.dim_input = self._get_input_dim()

    def _decouple_params(self):
        self.inter_name = self.dic_traffic_env_conf["INTER_NAME"]
        self._alpha = self.dic_agent_conf['ALPHA']
        self._min_alpha = self.dic_agent_conf['MIN_ALPHA']
        self._alpha_decay_rate = self.dic_agent_conf['ALPHA_DECAY_RATE']
        self._alpha_decay_step = self.dic_agent_conf['ALPHA_DECAY_STEP']
        self._norm = self.dic_agent_conf['NORM']
        self._num_updates = self.dic_agent_conf['NUM_UPDATES']

    # TODO complete and use this function
    def _get_input_dim(self):
        dim_input = 0
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dim_input += \
                len(self.dic_traffic_env_conf[
                        "LANE_PHASE_INFOS"][self.inter_name][
                        "start_lane"])
        return dim_input

    def decay_epsilon(self, round_number):
        """Warning: MODIFIED DIC_AGENT_CONF : EPSILON VALUE
        When reached round i = (log(eps_min)-log(eps_init)) / log(decay)
        eps_init reached eps_min. default round: 27
        """
        decayed_epsilon = \
            self.dic_agent_conf["EPSILON"] * \
            np.power(self.dic_agent_conf["EPSILON_DECAY"], round_number)
        self.dic_agent_conf["EPSILON"] = max(
            decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def decay_lr(self, round_number):
        """Warning: MODIFIED DIC_AGENT_CONF : LEARNING_RATE VALUE
        default: Not changed
        """
        decayed_lr = self.dic_agent_conf["LEARNING_RATE"] * pow(
            self.dic_agent_conf["LR_DECAY"], round_number)
        self.dic_agent_conf["LEARNING_RATE"] = max(decayed_lr,
                                                   self.dic_agent_conf[
                                                       "MIN_LR"])

    def choose_action(self, state, choice_random):
        raise NotImplementedError

    def build_network(self):
        raise NotImplementedError

    def load_network(self, file_name):
        raise NotImplementedError

    def load_network_bar(self, file_name):
        raise NotImplementedError

    def prepare_Xs_Y(self, memory):
        raise NotImplementedError

    def train_network(self):
        raise NotImplementedError

    def save_network(self, file_name):
        raise NotImplementedError
