import copy
import os
import pickle
import random
import numpy as np
from algs.agent_discrete import AgentDiscrete
from configs.conf_path import ConfPath


class GSQLAgent(AgentDiscrete):
    def __init__(self, conf_path: ConfPath, round_number: int, inter_name: str):
        self.conf_path = conf_path
        self.round_number = round_number
        self.inter_name = inter_name

        self.conf_exp, self.conf_agent, self.conf_traffic = \
            self.conf_path.load_conf_file(inter_name=self.inter_name)
        self.traffic_info = self.conf_traffic.TRAFFIC_INFO

        self.state_particle = self.conf_agent["PARTICLE"]
        self.epsilon = self.conf_agent["MIN_EPSILON"]
        self.discount = self.conf_agent["GAMMA"]
        self.learning_rate = self.conf_agent["LR"]
        self.w = self.conf_agent["W"]

        self.state_dim = self.state_particle ** len(self.traffic_info['list_lane_enters'])
        self.action_dim = len(self.traffic_info['phase_lane_mapping'])

        if self.round_number == 0:
            self.q_metrix = np.zeros((self.state_dim, self.action_dim))
            self.q_metrix_k = np.zeros((self.state_dim, self.action_dim))
            self.q_metrix_k1 = np.zeros((self.state_dim, self.action_dim))
        else:
            self.q_metrix = self.load_metrix(self.round_number - 1)
            self.q_metrix_k = copy.deepcopy(self.q_metrix)
            self.q_metrix_k1 = copy.deepcopy(self.q_metrix)

    def convert_state_to_input(self, state):
        list_vehicle = state['lane_vehicle_cnt']
        delta = 30 / self.state_particle
        state_out = 0
        for idx, vehicle_num in enumerate(list_vehicle):
            i_max = 0
            for i in range(self.state_particle):
                if vehicle_num > i * delta:
                    i_max = i
            state_out += (self.state_particle ** idx * i_max)
        return state_out

    def choose_action(self, state, choice_random=True):
        self.state = self.convert_state_to_input(state)
        action_value = self.q_metrix[self.state, :]
        if random.random() > self.epsilon:
            self.action = action_value.argmax()
        else:
            self.action = random.choice(range(self.action_dim))
        return self.action

    def save_metrix(self, round_number):
        file_name = os.path.join(self.conf_path.MODEL,
                                 'round_%d.pkl' % round_number)
        pickle.dump(self.q_metrix, open(file_name, mode='wb'))
        pass

    def load_metrix(self, round_number):
        file_name = os.path.join(self.conf_path.MODEL,
                                 'round_%d.pkl' % round_number)
        q_metrix = pickle.load(open(file_name, mode='rb'))
        return q_metrix

    def train_metrix(self, s, a, r, ns):
        s = self.convert_state_to_input(s)
        ns = self.convert_state_to_input(ns)

        temp1 = self.w * r + self.discount * self.w * \
                self.q_metrix_k1[ns, :].max() + (1 - self.w) * \
                self.q_metrix_k1[s, :].max()
        temp2 = self.w * r + self.discount * self.w * \
                self.q_metrix_k[ns, :].max() + (1 - self.w) * \
                self.q_metrix_k[s, :].max()

        self.q_metrix[s, a] = self.q_metrix_k[s, a] + self.learning_rate * \
                              (temp1 - self.q_metrix_k[s, a]) + (1 - self.learning_rate) * \
                              (temp2 - temp1)
        self.q_metrix_k1 = copy.deepcopy(self.q_metrix_k)
        self.q_metrix_k = copy.deepcopy(self.q_metrix)
