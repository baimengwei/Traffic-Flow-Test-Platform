import os.path
from algs.DQN.dqn_agent import DQN
from algs.agent import Agent
from configs.config_phaser import *
import torch
import numpy as np
import random


class METADQNAgent(Agent):
    def __init__(self, conf_path, round_number, inter_name,
                 list_traffic_name=None):
        super().__init__(conf_path, round_number, inter_name)

        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__inter_name = inter_name
        self.__list_traffic_name = list_traffic_name
        # a patch
        if 'generator' in self.__conf_path.WORK_SAMPLE:
            config_dir = os.path.join(self.__conf_path.WORK_SAMPLE, '..')
        else:
            config_dir = os.path.join(self.__conf_path.WORK_SAMPLE)
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file(
                config_dir=config_dir, inter_name=inter_name)
        self.__conf_agent = self.decay_epsilon(
            self.__conf_agent, self.__round_number)

        if self.__round_number == 0:
            self.build_network()
            self.load_network_bar(self.__list_traffic_name)
        else:
            self.load_network(self.__round_number - 1)
            self.load_network_bar(self.__list_traffic_name)

    def build_network(self):
        self.model = DQN(self.__conf_traffic)
        self.lossfunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def build_network_bar(self):
        print('use pretrained.')
        pass

    def load_network(self, round_number):
        """ used for meta
        """
        file_name = self.__inter_name + "_round_%d" % round_number + '.pt'
        file_path = os.path.join(self.__conf_path.MODEL, file_name)
        ckpt = torch.load(file_path)
        self.build_network()
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def load_network_bar(self, traffic_files):
        if traffic_files is None:
            return
        self.list_model_bar = []
        for traffic_file in traffic_files:
            file_path = os.path.join(
                self.__conf_path.MODEL, '..', '..', 'MODEL_DQN', traffic_file)
            file_path = os.path.join(file_path, os.listdir(file_path)[0])
            ckpt = torch.load(file_path)
            model_target = DQN(self.__conf_traffic)
            model_target.load_state_dict((ckpt['state_dict']))
            self.list_model_bar.append(model_target)

    def choose_action(self, state, choice_random=True):
        input = self.convert_state_to_input(self.__conf_traffic, state)
        input = torch.Tensor(input).flatten(0).unsqueeze(0)

        q_values = self.model.forward(input)
        if random.random() <= self.__conf_agent["EPSILON"] and choice_random:
            actions = random.randrange(len(q_values[0]))
        else:
            actions = np.argmax(q_values[0].detach().numpy())
        return actions

    def prepare_Xs_Y(self, sample_set, traffic_file_idx=None):
        if traffic_file_idx is None:
            raise ValueError('traffic_file should be input')
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
        q_values_bar = self.list_model_bar[traffic_file_idx].forward(
            torch.Tensor(next_state)).detach().numpy()
        reward_avg = np.array(reward_avg) / self.__conf_traffic.NORMAL_FACTOR
        gamma = self.__conf_agent['GAMMA']
        range_idx = list(range(len(q_values)))
        q_values[range_idx, action] = \
            reward_avg + gamma * np.max(q_values_bar, axis=-1)
        if 'Xs' not in self.__dict__.keys():
            self.Xs = np.array(state)
            self.Y = q_values
        else:
            self.Xs = np.vstack((self.Xs, np.array(state)))
            self.Y = np.vstack((self.Y, q_values))

    def train_network(self):
        """used for each task
        """
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
        """
        """
        file_path = os.path.join(
            self.__conf_path.MODEL,
            self.__inter_name + '_round_%d' % round_number + '.pt')
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lossfunc': self.lossfunc.state_dict()}
        torch.save(ckpt, file_path)
