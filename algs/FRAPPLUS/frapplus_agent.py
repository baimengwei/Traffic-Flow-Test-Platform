import os
import random
import numpy as np
from algs.agent import Agent
import torch
import torch.nn as nn


class FRAPPLUS(nn.Module):
    def __init__(self, conf_traffic):
        super().__init__()
        self.__conf_traffic = conf_traffic

        self.traffic_info = self.__conf_traffic.TRAFFIC_INFO

        self.list_lane = self.traffic_info['list_lane_enters']
        self.list_phase = list(self.traffic_info['phase_lane_mapping'].keys())
        self.phase_links = self.traffic_info['phase_links']
        self.constant_mask = torch.Tensor(self.traffic_info['relation']).int()

        self.phase_dim = len(self.traffic_info['phase_links'])
        self.vehicle_dim = len(self.traffic_info['phase_links'])

        self.embeding_phase = nn.Embedding(2, 4)
        self.activate_phase = nn.Sigmoid()
        self.embeding_vehicle = nn.Linear(1, 4)
        # torch.nn.init.uniform_(self.embeding_vehicle.weight, -0.05, 0.05)
        self.activate_vehicle = nn.Sigmoid()

        self.weight_feature_line = torch.nn.Linear(8, 16)
        self.activate_feature_line = torch.nn.ReLU()
        # torch.nn.init.uniform_(self.weight_feature_line.weight)

        self.embeding_constant = nn.Embedding(2, 4)

        self.conv_feature = torch.nn.Conv2d(32, 20, 1)
        self.activate_conv_feature = torch.nn.ReLU()
        self.conv_constant = torch.nn.Conv2d(4, 20, 1)
        self.activate_conv_constant = torch.nn.ReLU()
        self.conv_combine = torch.nn.Conv2d(20, 20, 1)
        self.activate_conv_combine = torch.nn.ReLU()
        self.conv_final = torch.nn.Conv2d(20, 1, 1)

    def forward(self, feature_input):
        batch_size = feature_input.shape[0]
        feature_phase = feature_input[:, :self.phase_dim]
        feature_vehicle = feature_input[:, self.vehicle_dim:]

        feature_phase = self.embeding_phase(feature_phase.int())
        feature_phase = self.activate_phase(feature_phase)
        dic_feature_lane = {}
        for idx, lane in enumerate(self.list_lane):
            tmp_veh = self.embeding_vehicle(
                feature_vehicle[:, idx].reshape(batch_size, 1))
            tmp_veh = self.activate_vehicle(tmp_veh)
            tmp_phase = feature_phase[:, idx, ]
            dic_feature_lane[lane] = torch.cat((tmp_veh, tmp_phase), dim=-1)
        # # For FRAPPlus
        # list_phase_pressure = []
        # for phase_index in self.list_phase:
        #     lanes = self.phase_links[phase_index]
        #     phase_pressure = torch.zeros((batch_size, 16))
        #     for line in lanes:
        #         tmp = self.weight_feature_line(dic_feature_lane[line])
        #         tmp = self.activate_feature_line(tmp)
        #         phase_pressure += tmp
        #     count = len(self.phase_links[phase_index])
        #     list_phase_pressure.append(phase_pressure/count)

        list_phase_pressure = []
        for phase_index in self.list_phase:
            lane_map = self.traffic_info['phase_lane_mapping'][phase_index]
            list_lane_enters = self.traffic_info['list_lane_enters']
            lane_combine = []
            for idx,l in enumerate(lane_map):
                if l == 1:
                    lane_combine.append(idx)
            lane1 = list_lane_enters[lane_combine[0]]
            lane2 = list_lane_enters[lane_combine[0]]

            lane1 = self.weight_feature_line(dic_feature_lane[lane1])
            lane1 = self.activate_feature_line(lane1)
            lane2 = self.weight_feature_line(dic_feature_lane[lane2])
            lane2 = self.activate_feature_line(lane2)
            combine = lane1 + lane2
            list_phase_pressure.append(combine)

        constant_mask = torch.tile(self.constant_mask, (batch_size, 1, 1))
        constant_mask = self.embeding_constant(constant_mask)
        constant_mask = torch.transpose(constant_mask, 3, 1)
        constant_mask = torch.transpose(constant_mask, 3, 2)

        list_phase_pressure_matrix = []
        num_phase = len(self.list_phase)
        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    phase_con = torch.cat([list_phase_pressure[i],
                                           list_phase_pressure[j]],
                                          dim=-1)
                    list_phase_pressure_matrix.append(phase_con)
        feature_mask = torch.cat(list_phase_pressure_matrix, dim=-1).reshape(
            batch_size, num_phase, num_phase - 1, 32)
        feature_mask = torch.transpose(feature_mask, 3, 1)
        feature_mask = torch.transpose(feature_mask, 3, 2)

        x = self.conv_feature(feature_mask)
        x = self.activate_conv_feature(x)
        y = self.conv_constant(constant_mask)
        y = self.activate_conv_constant(y)
        z = x * y
        z = self.conv_combine(z)
        z = self.activate_conv_combine(z)
        z = self.conv_final(z)
        z = torch.sum(z, dim=-1)
        z = z.reshape((-1, num_phase))
        return z


class FRAPPLUSAgent(Agent):
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
        self.model = FRAPPLUS(self.__conf_traffic)
        self.lossfunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def build_network_bar(self):
        self.model_target = FRAPPLUS(self.__conf_traffic)
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
        self.model_target = FRAPPLUS(self.__conf_traffic)
        self.model_target.load_state_dict((ckpt['state_dict']))

    def save_network(self, round_number):
        file_path = os.path.join(
            self.__conf_path.MODEL,
            self.__inter_name + '_round_%d' % round_number + '.pt')
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lossfunc': self.lossfunc.state_dict()}
        torch.save(ckpt, file_path)

    def prepare_Xs_Y(self, sample_set):
        state = []
        action = []
        next_state = []
        reward_avg = []
        # TODO temp use, need to modify
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

    def choose_action(self, state, choice_random=True):
        """
        """
        input = self.convert_state_to_input(self.__conf_traffic, state)
        input = torch.Tensor(input).flatten(0).unsqueeze(0)
        q_values = self.model.forward(input)
        if random.random() <= self.__conf_agent["EPSILON"] and choice_random:
            actions = random.randrange(len(q_values[0]))
        else:
            actions = np.argmax(q_values[0].detach().numpy())
        return actions
