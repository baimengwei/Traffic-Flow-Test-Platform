import os
import random
import numpy as np
import torch
import torch.nn as nn
from algs.agent import Agent


class Context(nn.Module):
    def __init__(self, conf_traffic):
        """
        """
        super().__init__()
        self.__conf_traffic = conf_traffic

        self.traffic_info = self.__conf_traffic.TRAFFIC_INFO
        phase_dim = len(self.traffic_info['phase_links'])
        vehicle_dim = len(self.traffic_info['phase_links'])
        self.state_dim = phase_dim + vehicle_dim

        self.action_dim = len(self.traffic_info['phase_lane_mapping'][0])  # one hot represent according to phase.
        self.input_dim = self.action_dim + 1 + self.state_dim
        self.hidden_dim = 10

        self.recurrent = nn.GRU(self.input_dim, self.hidden_dim,
                                bidirectional=False, batch_first=True,
                                num_layers=1)

    def forward(self, history_input):
        """
        hidden_dim origin: (layer_dim, batch_size, hidden_size)
        """
        batch_size = len(history_input)
        hidden = torch.zeros(1, batch_size, self.hidden_dim)
        try:
            _, hidden = self.recurrent(history_input, hidden)
        except:
            print(1)
            raise ValueError('error in rnn')
        out = hidden.squeeze(0)
        return out


class DRQN(nn.Module):
    def __init__(self, conf_traffic):
        super().__init__()
        self.__conf_traffic = conf_traffic

        self.traffic_info = self.__conf_traffic.TRAFFIC_INFO
        phase_dim = len(self.traffic_info['phase_links'])
        vehicle_dim = len(self.traffic_info['phase_links'])
        self.state_dim = phase_dim + vehicle_dim
        self.action_dim = len(self.traffic_info['phase_lane_mapping'])
        #
        self.hidden_dim = 10
        self.weight_feature_line = torch.nn.Linear(
            self.state_dim + self.hidden_dim, 50)
        self.activate_feature_line = torch.nn.ReLU()

        self.linear_combine = torch.nn.Linear(50, 50)
        self.activate_linear_combine = torch.nn.ReLU()
        self.linear_final = torch.nn.Linear(50, self.action_dim)

        self.rnn_layer = Context(self.__conf_traffic)

    def forward(self, feature_input, history_input):
        history_output = self.rnn_layer(history_input)
        combine = torch.cat((feature_input, history_output), dim=-1)
        combine = self.weight_feature_line(combine)
        combine = self.activate_feature_line(combine)
        combine = self.linear_combine(combine)
        combine = self.activate_linear_combine(combine)
        combine = self.linear_final(combine)
        return combine


class DRQNAgent(Agent):
    def __init__(self, conf_path, round_number, inter_name):
        super().__init__(conf_path, round_number, inter_name)
        self.__conf_path = conf_path
        self.__round_number = round_number
        self.inter_name = inter_name
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file(inter_name=inter_name)
        self.__conf_agent = self.decay_epsilon(
            self.__conf_agent, self.__round_number)

        self.list_history = []

        if self.__round_number == 0:
            self.build_network()
            self.build_network_bar()
        else:
            self.load_network(self.__round_number - 1)
            bar_freq = self.__conf_agent["UPDATE_Q_BAR_FREQ"]
            bar_number = (self.__round_number - 1) // bar_freq * bar_freq
            self.load_network_bar(bar_number)

    def build_network(self):
        self.model = DRQN(self.__conf_traffic)
        self.lossfunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def build_network_bar(self):
        self.model_target = DRQN(self.__conf_traffic)
        self.model_target.load_state_dict(self.model.state_dict())

    def load_network(self, round_number):
        file_name = self.inter_name + "_round_%d" % round_number + '.pt'
        file_path = os.path.join(self.__conf_path.MODEL, file_name)
        ckpt = torch.load(file_path)
        self.build_network()
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def load_network_bar(self, round_number):
        file_name = self.inter_name + "_round_%d" % round_number + '.pt'
        file_path = os.path.join(self.__conf_path.MODEL, file_name)
        ckpt = torch.load(file_path)
        self.model_target = DRQN(self.__conf_traffic)
        self.model_target.load_state_dict((ckpt['state_dict']))

    def choose_action(self, state, history_input, choice_random=True):
        input = self.convert_state_to_input(self.__conf_traffic, state)
        input = torch.Tensor(input).flatten(0).unsqueeze(0)
        history = torch.Tensor(history_input).unsqueeze(0)
        q_values = self.model.forward(input, history)
        if random.random() <= self.__conf_agent["EPSILON"] and choice_random:
            actions = random.randrange(len(q_values[0]))
        else:
            actions = np.argmax(q_values[0].detach().numpy())
        self.list_history.append(np.array(history_input).tolist())
        return actions

    def prepare_Xs_Y(self, sample_set):
        state = []
        action = []
        next_state = []
        reward_avg = []
        history_input = []
        for each in sample_set:
            state.append(each[0]['cur_phase_index'] + each[0]['lane_vehicle_cnt'])
            action.append(each[1])
            next_state.append(
                each[2]['cur_phase_index'] + each[2]['lane_vehicle_cnt'])
            reward_avg.append(each[3])
            history_input.append(each[5])

        q_values = \
            self.model.forward(torch.Tensor(state), torch.Tensor(history_input)
                               ).detach().numpy()
        q_values_bar = \
            self.model_target.forward(torch.Tensor(next_state),
                                      torch.Tensor(history_input)
                                      ).detach().numpy()
        reward_avg = np.array(reward_avg) / self.__conf_traffic.NORMAL_FACTOR
        gamma = self.__conf_agent["GAMMA"]
        range_idx = list(range(len(q_values)))
        q_values[range_idx, action] = \
            reward_avg + gamma * np.max(q_values_bar, axis=-1)
        self.Xs = np.array(state), np.array(history_input)
        self.Y = q_values

    def train_network(self):
        epochs = self.__conf_agent["EPOCHS"]
        for i in range(epochs):
            sample_x, history_input = self.Xs
            sample_y = self.Y
            yp = self.model.forward(torch.Tensor(sample_x),
                                    torch.Tensor(history_input))
            loss = self.lossfunc(yp, torch.Tensor(sample_y))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print('%d memory, updating... %d, loss: %.4f'
            #       % (len(self.Y), i, loss.item()))

    def save_network(self, round_number):
        file_path = os.path.join(
            self.__conf_path.MODEL,
            self.inter_name + '_round_%d' % round_number + '.pt')
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lossfunc': self.lossfunc.state_dict()}
        torch.save(ckpt, file_path)
