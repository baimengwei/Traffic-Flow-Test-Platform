import os
import random

import numpy as np
import torch
import torch.nn as nn
from algs.agent import Agent


class DQN(nn.Module):
    def __init__(self, conf_traffic):
        super().__init__()
        self.__conf_traffic = conf_traffic

        self.traffic_info = self.__conf_traffic.TRAFFIC_INFO
        phase_dim = len(self.traffic_info['phase_links'])
        vehicle_dim = len(self.traffic_info['phase_links'])
        self.state_dim = phase_dim + vehicle_dim
        self.action_dim = len(self.traffic_info['phase_lane_mapping'])

        self.weight_feature_line = torch.nn.Linear(self.state_dim, 50)
        self.activate_feature_line = torch.nn.ReLU()

        self.linear_combine = torch.nn.Linear(50, 50)
        self.activate_linear_combine = torch.nn.ReLU()
        self.linear_final = torch.nn.Linear(50, self.action_dim)

    def forward(self, feature_input):
        combine = self.weight_feature_line(feature_input)
        combine = self.activate_feature_line(combine)
        combine = self.linear_combine(combine)
        combine = self.activate_linear_combine(combine)
        combine = self.linear_final(combine)
        return combine


class DQNAgent(Agent):
    def __init__(self, conf_path, round_number, inter_name):
        super().__init__(conf_path, round_number, inter_name)

        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__inter_name = inter_name
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file(inter_name=inter_name)
        self.__conf_agent = self.decay_epsilon(
            self.__conf_agent, self.__round_number)

        if self.__round_number == 0:
            self.build_network()
            self.build_network_bar()
        else:
            self.load_network(self.__round_number - 1)
            bar_freq = self.__conf_agent["UPDATE_Q_BAR_FREQ"]
            bar_number = (self.__round_number - 1) // bar_freq * bar_freq
            self.load_network_bar(bar_number)

    def build_network(self):
        self.model = DQN(self.__conf_traffic)
        self.lossfunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # FOR META bellow, modify the code by hand.
        # import warnings
        # warnings.warn('loading meta parameters...')
        # meta_network = os.path.join(
        #     self.__conf_path.MODEL, '..', '..', 'intersection_1_1_round_99.pt')
        # ckpt = torch.load(meta_network)
        # self.model.load_state_dict(ckpt['state_dict'])
        # self.optimizer.load_state_dict(ckpt['optimizer'])
        # self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def build_network_bar(self):
        self.model_target = DQN(self.__conf_traffic)
        self.model_target.load_state_dict(self.model.state_dict())

    def load_network(self, round_number):
        file_name = self.__inter_name + "_round_%d" % round_number + '.pt'
        file_path = os.path.join(self.__conf_path.MODEL, file_name)
        ckpt = torch.load(file_path)
        self.build_network()
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def load_network_bar(self, round_number):
        file_name = self.__inter_name + "_round_%d" % round_number + '.pt'
        file_path = os.path.join(self.__conf_path.MODEL, file_name)
        ckpt = torch.load(file_path)
        self.model_target = DQN(self.__conf_traffic)
        self.model_target.load_state_dict((ckpt['state_dict']))

    def choose_action(self, state, choice_random=True):
        input = self.convert_state_to_input(self.__conf_traffic, state)
        input = torch.Tensor(input).flatten(0).unsqueeze(0)

        q_values = self.model.forward(input)
        if random.random() <= self.__conf_agent["EPSILON"] and choice_random:
            actions = random.randrange(len(q_values[0]))
        else:
            actions = np.argmax(q_values[0].detach().numpy())
        return actions

    def prepare_Xs_Y(self, sample_set):
        state = []
        action = []
        next_state = []
        reward_avg = []
        for each in sample_set:
            state.append(each[0]['cur_phase_index'] + each[0]['lane_vehicle_cnt'])
            action.append(each[1])
            next_state.append(
                each[2]['cur_phase_index'] + each[2]['lane_vehicle_cnt'])
            reward_avg.append(each[3])
        q_values = self.model.forward(torch.Tensor(state)).detach().numpy()
        q_values_bar = self.model_target.forward(
            torch.Tensor(next_state)).detach().numpy()
        reward_avg = np.array(reward_avg) / self.__conf_traffic.NORMAL_FACTOR
        gamma = self.__conf_agent['GAMMA']
        range_idx = list(range(len(q_values)))
        q_values[range_idx, action] = \
            reward_avg + gamma * np.max(q_values_bar, axis=-1)
        self.Xs = np.array(state)
        self.Y = q_values

    def train_network(self):
        epochs = self.__conf_agent["EPOCHS"]
        for i in range(epochs):
            sample_x = self.Xs
            sample_y = self.Y
            yp = self.model.forward(torch.Tensor(sample_x))
            loss = self.lossfunc(yp, torch.Tensor(sample_y))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print('%d memory, updating... %d, loss: %.4f'
            #       % (len(self.Y), i, loss.item()))

    def save_network(self, round_number):
        file_path = os.path.join(
            self.__conf_path.MODEL,
            self.__inter_name + '_round_%d' % round_number + '.pt')
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lossfunc': self.lossfunc.state_dict()}
        torch.save(ckpt, file_path)
