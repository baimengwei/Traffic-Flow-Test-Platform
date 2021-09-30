import os
import random
import numpy as np

from algs.DQN.dqn_agent import DQNAgent


class DQNAdaptAgent(DQNAgent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

        self.decay_epsilon(self.round_number)

        if self.round_number == 0:
            self.load_network('../META_A/round_meta')
            self.load_network_bar('../META_A/round_meta')
        else:
            self.load_network("round_%d" % (self.round_number - 1))
            bar_freq = self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
            bar_number = (self.round_number - 1) // bar_freq * bar_freq
            self.load_network_bar("round_%d" % bar_number)

