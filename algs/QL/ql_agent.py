import os
import pickle
import random

import numpy as np

from algs.agent_discrete import AgentDiscrete
from configs.conf_path import ConfPath


class QLAgent(AgentDiscrete):
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

        self.state_dim, self.lane_enters_once = self.get_state_dim()
        self.action_dim = len(self.traffic_info['phase_lane_mapping'])
        if self.round_number == 0:
            self.q_metrix = np.zeros((self.state_dim, self.action_dim))
        else:
            self.q_metrix = self.load_metrix(self.round_number - 1)

    def get_state_dim(self):
        lane_enters = self.traffic_info['list_lane_enters']
        lane_enters_once = []
        lane_enters_name = []
        for lane in lane_enters:
            if lane not in lane_enters_name:
                lane_enters_once.append(True)
                lane_enters_name.append(lane)
            else:
                lane_enters_once.append(False)
        state_dim = self.conf_agent['PARTICLE'] ** sum(lane_enters_once)
        return state_dim, lane_enters_once

    def convert_state_to_input(self, state):
        list_vehicle = state['lane_vehicle_cnt']
        list_vehicle_ = []
        for each_lv, flag in zip(list_vehicle, self.lane_enters_once):
            if flag is True:
                list_vehicle_.append(each_lv)
        list_vehicle = list_vehicle_

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
        file_name = os.path.join(
            self.conf_path.MODEL,
            'round_%d_%s.pkl' % (round_number, self.inter_name))
        pickle.dump(self.q_metrix, open(file_name, mode='wb'))
        pass

    def load_metrix(self, round_number):
        file_name = os.path.join(
            self.conf_path.MODEL,
            'round_%d_%s.pkl' % (round_number, self.inter_name))
        return pickle.load(open(file_name, mode='rb'))

    def train_metrix(self, s, a, r, ns):
        s_ = self.convert_state_to_input(s)
        ns_ = self.convert_state_to_input(ns)

        delta = r + self.discount * \
                self.q_metrix[ns_, :].max() - self.q_metrix[s_, a]
        delta_q = self.learning_rate * delta

        self.q_metrix[s_, a] = self.q_metrix[s_, a] + delta_q
