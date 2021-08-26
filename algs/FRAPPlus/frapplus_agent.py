import os
import random
import numpy as np
from algs.agent import Agent
import torch
import torch.nn as nn


class FRAP(nn.Module):
    def __init__(self, dic_traffic_env_conf):
        super().__init__()

        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.line_phase_info = dic_traffic_env_conf["LANE_PHASE_INFO"]

        self.list_lane = self.line_phase_info['start_lane']
        self.list_phase = self.line_phase_info['phase']
        self.phase_line_mapping = \
            self.line_phase_info['phase_startLane_mapping']
        self.constant_mask = \
            torch.Tensor(self.line_phase_info['relation']).int()

        dim_feature = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]
        self.phase_dim = dim_feature['cur_phase'][0]
        self.vehicle_dim = dim_feature['lane_num_vehicle'][0]

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
        #     lanes = self.phase_line_mapping[phase_index]
        #     phase_pressure = torch.zeros((batch_size, 16))
        #     for line in lanes:
        #         tmp = self.weight_feature_line(dic_feature_lane[line])
        #         tmp = self.activate_feature_line(tmp)
        #         phase_pressure += tmp
        #     count = len(self.phase_line_mapping[phase_index])
        #     list_phase_pressure.append(phase_pressure/count)

        list_phase_pressure = []
        for phase_index in self.list_phase:
            line_combine = self.phase_line_mapping[phase_index]
            line1, line2 = line_combine[0], line_combine[1]
            line1 = self.weight_feature_line(dic_feature_lane[line1])
            line2 = self.weight_feature_line(dic_feature_lane[line2])
            combine = line1 + line2
            list_phase_pressure.append(combine)

        constant_mask = self.embeding_constant(self.constant_mask)
        constant_mask = torch.tile(constant_mask.unsqueeze(0),
                                   (batch_size, 1, 1, 1))
        constant_mask = torch.transpose(constant_mask, 3, 1)
        constant_mask = torch.transpose(constant_mask, 3, 2)

        feature_mask = torch.tensor([])
        num_phase = len(self.list_phase)
        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    phase_con = torch.cat([list_phase_pressure[i],
                                           list_phase_pressure[j]],
                                          dim=-1)
                    feature_mask = torch.cat([feature_mask, phase_con])

        feature_mask = feature_mask.reshape((-1, num_phase, num_phase - 1, 32))
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


class FRAPPlusAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path,
                         round_number)

        if self.round_number == 0:
            self.build_network()
            self.build_network_bar()
        else:
            self.load_network("round_%d" % (self.round_number - 1))
            bar_freq = self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
            bar_number = (self.round_number - 1) // bar_freq * bar_freq
            self.load_network_bar("round_%d" % bar_number)

    def build_network(self):
        self.model = FRAP(self.dic_traffic_env_conf)
        self.lossfunc = torch.nn.MSELoss()
        self.optimizer = \
            torch.optim.Adam(self.model.parameters(),
                             lr=self.dic_agent_conf["LEARNING_RATE"])

    def build_network_bar(self):
        self.model_target = FRAP(self.dic_traffic_env_conf)
        self.model_target.load_state_dict(self.model.state_dict())

    def load_network(self, file_name):
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = torch.load(file_path)
        self.model = FRAP(self.dic_traffic_env_conf)
        self.model.load_state_dict(ckpt['state_dict'])

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lossfunc = torch.nn.MSELoss()
        self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def load_network_bar(self, file_name):
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = torch.load(file_path)
        self.model_target = FRAP(self.dic_traffic_env_conf)
        self.model_target.load_state_dict((ckpt['state_dict']))

    def save_network(self, file_name):
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
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
            state.append(each[0]['cur_phase'] + each[0]['lane_num_vehicle'])
            action.append(each[1])
            next_state.append(
                each[2]['cur_phase'] + each[2]['lane_num_vehicle'])
            reward_avg.append(each[3])

        q_values = self.model.forward(torch.Tensor(state)).detach().numpy()
        q_values_bar = self.model_target.forward(
            torch.Tensor(next_state)).detach().numpy()
        reward_avg = np.array(reward_avg) / self.dic_agent_conf["NORMAL_FACTOR"]
        gamma = self.dic_agent_conf["GAMMA"]
        range_idx = list(range(len(q_values)))
        q_values[range_idx, action] = \
            reward_avg + gamma * np.max(q_values_bar, axis=-1)
        self.Xs = np.array(state)
        self.Y = q_values

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
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
        inputs = self.convert_state_to_input(state)
        inputs = torch.Tensor(inputs).flatten().unsqueeze(0)
        q_values = self.model.forward(inputs)
        if random.random() <= self.dic_agent_conf["EPSILON"] \
                and choice_random:
            actions = random.randrange(len(q_values[0]))
        else:
            actions = np.argmax(q_values[0].detach().numpy())
        return actions

    def convert_state_to_input(self, s):
        inputs = []
        dic_phase_expansion = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]['phase_map']
        for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature == "cur_phase":
                inputs.append(np.array([dic_phase_expansion[s[feature][0]]]))
            else:
                inputs.append(np.array([s[feature]]))
        return inputs
