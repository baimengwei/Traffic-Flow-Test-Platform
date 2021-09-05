import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from algs.agent import Agent


class Context(nn.Module):
    def __init__(self, dic_traffic_env_conf, dic_agent_conf):
        """
        """
        super().__init__()
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.lane_phase_info = dic_traffic_env_conf["LANE_PHASE_INFO"]

        dim_feature = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]
        phase_dim = dim_feature['cur_phase'][0]
        vehicle_dim = dim_feature['lane_num_vehicle'][0]
        self.state_dim = phase_dim + vehicle_dim
        self.action_dim = phase_dim  # one hot represent according to phase.
        self.input_dim = self.action_dim + 1 + self.state_dim
        self.hidden_dim = self.dic_agent_conf["HIDDEN_DIM"]

        self.recurrent = nn.GRU(self.input_dim, self.hidden_dim,
                                bidirectional=False, batch_first=True,
                                num_layers=1)

    def forward(self, history_input):
        """
        hidden_dim origin: (layer_dim, batch_size, hidden_size)
        """
        batch_size = len(history_input)
        hidden = torch.zeros(1, batch_size, self.hidden_dim)
        _, hidden = self.recurrent(history_input, hidden)
        out = hidden.squeeze(0)
        return out


class DRQN(nn.Module):
    def __init__(self, dic_traffic_env_conf, dic_agent_conf):
        super().__init__()
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.lane_phase_info = dic_traffic_env_conf["LANE_PHASE_INFO"]

        dim_feature = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]
        phase_dim = dim_feature['cur_phase'][0]
        vehicle_dim = dim_feature['lane_num_vehicle'][0]
        self.state_dim = phase_dim + vehicle_dim
        self.action_dim = len(self.lane_phase_info['phase'])
        self.hidden_dim = self.dic_agent_conf["HIDDEN_DIM"]

        self.weight_feature_line = torch.nn.Linear(
            self.state_dim + self.hidden_dim, 100)
        self.activate_feature_line = torch.nn.ReLU()

        self.linear_combine = torch.nn.Linear(100, 100)
        self.activate_linear_combine = torch.nn.ReLU()
        self.linear_final = torch.nn.Linear(100, self.action_dim)

        self.rnn_layer = Context(self.dic_traffic_env_conf, dic_agent_conf)

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
    def __init__(self, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path,
                         round_number)
        self.decay_epsilon(self.round_number)
        self.list_history = []

        if self.round_number == 0:
            self.build_network()
            self.build_network_bar()
        else:
            self.load_network("round_%d" % (self.round_number - 1))
            bar_freq = self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
            bar_number = (self.round_number - 1) // bar_freq * bar_freq
            self.load_network_bar("round_%d" % bar_number)

    def build_network(self):
        self.model = DRQN(self.dic_traffic_env_conf, self.dic_agent_conf)
        self.lossfunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.dic_agent_conf["LR"])

    def build_network_bar(self):
        self.model_target = DRQN(self.dic_traffic_env_conf, self.dic_agent_conf)
        self.model_target.load_state_dict(self.model.state_dict())

    def load_network(self, file_name):
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = torch.load(file_path)
        self.build_network()
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def load_network_bar(self, file_name):
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = torch.load(file_path)
        self.model_target = DRQN(self.dic_traffic_env_conf, self.dic_agent_conf)
        self.model_target.load_state_dict((ckpt['state_dict']))

    def choose_action(self, state, history_input, choice_random=True):
        state = self.convert_state_to_input(state)
        input = torch.Tensor(state).flatten().unsqueeze(0)
        history = torch.Tensor(history_input).unsqueeze(0)
        q_values = self.model.forward(input, history)
        if random.random() <= self.dic_agent_conf["EPSILON"] \
                and choice_random:
            actions = random.randrange(len(q_values[0]))
        else:
            actions = np.argmax(q_values[0].detach().numpy())

        self.list_history.append(np.array(history_input).tolist())
        return actions

    def convert_state_to_input(self, s):
        input = []
        dic_phase_expansion = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]['phase_map']
        for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature == "cur_phase":
                input.append(np.array([dic_phase_expansion[s[feature][0]]]))
            else:
                input.append(np.array([s[feature]]))
        return input

    def prepare_Xs_Y(self, sample_set):
        state = []
        action = []
        next_state = []
        reward_avg = []
        history_input = []
        for each in sample_set:
            state.append(each[0]['cur_phase'] + each[0]['lane_num_vehicle'])
            action.append(each[1])
            next_state.append(
                each[2]['cur_phase'] + each[2]['lane_num_vehicle'])
            reward_avg.append(each[3])
            history_input.append(each[5])

        q_values = \
            self.model.forward(torch.Tensor(state), torch.Tensor(history_input)
                               ).detach().numpy()
        q_values_bar = \
            self.model_target.forward(torch.Tensor(next_state),
                                      torch.Tensor(history_input)
                                      ).detach().numpy()
        reward_avg = np.array(reward_avg) / self.dic_agent_conf["NORMAL_FACTOR"]
        gamma = self.dic_agent_conf["GAMMA"]
        range_idx = list(range(len(q_values)))
        q_values[range_idx, action] = \
            reward_avg + gamma * np.max(q_values_bar, axis=-1)
        self.Xs = np.array(state), np.array(history_input)
        self.Y = q_values

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
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

    def save_network(self, file_name):
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lossfunc': self.lossfunc.state_dict()}
        torch.save(ckpt, file_path)

    def save_history(self):
        work_path = self.dic_path["PATH_TO_WORK"]
        for each_file in os.listdir(work_path):
            if ".pkl" in each_file:
                file_name = os.path.join(work_path, each_file)
                with open(file_name, "rb") as f:
                    logging_data = pickle.load(f)
                length_cnt = 0
                for each_data in logging_data:
                    if each_data["action"] != -1:
                        each_data["history"] = self.list_history[length_cnt]
                        length_cnt += 1
                if length_cnt != len(self.list_history):
                    raise ValueError(length_cnt, " vs ", len(self.list_history))
                with open(file_name, "wb") as f:
                    pickle.dump(logging_data, f)


