import os
import pickle
import random
import numpy as np
from algs.agent_discrete import AgentDiscrete
from configs.conf_path import ConfPath


class DynaMemory:
    def __init__(self, memory_length=2, memory_minibatch=1):
        self.memory_length = memory_length
        self.memory_minibatch = memory_minibatch
        self.state_size = 1
        self.action_size = 1
        self.memory_width = self.state_size * 2 + self.action_size + 1
        self.memory = np.zeros((self.memory_length, self.memory_width))
        self.index = 0
        self.max_index = 0

    def store(self, state, action, reward, next_state):
        transacton = np.hstack((state, action, reward, next_state))
        self.memory[self.index, :] = transacton

        self.index += 1
        if self.index % self.memory_length == 0:
            self.index = 0
        if self.max_index < self.memory_length:
            self.max_index += 1

    def sample(self):
        choice_random = np.random.choice(self.max_index, self.memory_minibatch)
        choice_data = self.memory[choice_random, :]
        state = choice_data[:, 0:self.state_size]
        action = choice_data[:, self.state_size:self.state_size + self.action_size]
        reward = choice_data[:, self.state_size + self.action_size:
                                self.state_size + self.action_size + 1]
        next_state = choice_data[:, self.state_size + self.action_size + 1:]

        reward = np.squeeze(reward)
        if self.action_size == 1:
            action = np.squeeze(action)
            action = int(action)
        if self.state_size == 1:
            state = np.squeeze(state)
            next_state = np.squeeze(next_state)
            state = int(state)
            next_state = int(next_state)

        return state, action, reward, next_state


class DYNAQAgent(AgentDiscrete):
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
        self.plan_cnt = self.conf_agent["PLAN_CNT"]

        self.state_dim, self.lane_enters_once = self.get_state_dim()
        self.action_dim = len(self.traffic_info['phase_lane_mapping'])

        if self.round_number == 0:
            self.q_metrix = np.zeros((self.state_dim, self.action_dim))
            self.q_memory = DynaMemory()
        else:
            self.q_metrix, self.q_memory = self.load_metrix(self.round_number - 1)

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
        file_name = os.path.join(self.conf_path.MODEL,
                                 'round_%d_%s.pkl' % (round_number, self.inter_name))
        pickle.dump(self.q_metrix, open(file_name, mode='wb'))
        file_name = os.path.join(self.conf_path.MODEL,
                                 'memory_%d_%s.pkl' % (round_number, self.inter_name))
        pickle.dump(self.q_memory, open(file_name, mode='wb'))
        pass

    def load_metrix(self, round_number):
        file_name = os.path.join(self.conf_path.MODEL,
                                 'round_%d_%s.pkl' % (round_number, self.inter_name))
        q_metrix = pickle.load(open(file_name, mode='rb'))
        file_name = os.path.join(self.conf_path.MODEL,
                                 'memory_%d_%s.pkl' % (round_number, self.inter_name))
        q_memory = pickle.load(open(file_name, mode='rb'))
        return q_metrix, q_memory

    def train_metrix(self, s, a, r, ns):
        s = self.convert_state_to_input(s)
        ns = self.convert_state_to_input(ns)
        self.q_memory.store(s, a, r, ns)

        delta = r + self.discount * \
                self.q_metrix[ns, :].max() - self.q_metrix[s, a]
        delta_q = self.learning_rate * delta
        self.q_metrix[s, a] = self.q_metrix[s, a] + delta_q

        # Reuse Memory
        for i in range(self.plan_cnt):
            model_state, model_action, model_reward, model_next_state = \
                self.q_memory.sample()
            delta = model_reward + self.discount * \
                    self.q_metrix[model_next_state, :].max() - \
                    self.q_metrix[model_state, model_action]
            delta_q = self.learning_rate * delta
            self.q_metrix[model_state, model_action] = \
                self.q_metrix[model_state, model_action] + delta_q
