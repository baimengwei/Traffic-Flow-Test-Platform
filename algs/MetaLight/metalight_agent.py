import os.path
import random
import shutil

import numpy as np
import torch
from algs.FRAPPlus.frapplus_agent import FRAPPlus
from algs.agent import Agent
from configs.config_phaser import *


class MetaLightAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 round_number, mode='task'):
        """mode is mata or task, which is supposed to define. default is 'task'
        """
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path,
                         round_number)
        self.decay_epsilon(self.round_number)
        self.mode = mode
        if self.round_number == 0 and self.mode == 'meta':
            self.build_network()
        elif self.mode == 'task':
            self.load_network("round_%d" % self.round_number)
            bar_freq = self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
            bar_number = self.round_number // bar_freq * bar_freq
            self.load_network_bar("round_%d" % bar_number)
        elif self.mode == 'meta':
            self.load_network_meta()
        else:
            raise NotImplementedError('a bug is here !')

    def build_network(self):
        """used for meta agent.
        """
        self.model_meta = FRAPPlus(self.dic_traffic_env_conf)
        self.lossfunc_meta = torch.nn.MSELoss()
        self.optimizer_meta = \
            torch.optim.Adam(self.model_meta.parameters(),
                             lr=self.dic_agent_conf["LR"])

    def build_network_bar(self):
        pass

    def choose_action(self, state, choice_random=True):
        """used for each task
        """
        input = self.convert_state_to_input(state)
        input = torch.Tensor(input).flatten().unsqueeze(0)
        q_values = self.model.forward(input)
        if random.random() <= self.dic_agent_conf["EPSILON"] \
                and choice_random:
            actions = random.randrange(len(q_values[0]))
        else:
            actions = np.argmax(q_values[0].detach().numpy())
        return actions

    def convert_state_to_input(self, s):
        """used for each task
        """
        input = []
        dic_phase_expansion = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]['phase_map']
        for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature == "cur_phase":
                input.append(np.array([dic_phase_expansion[s[feature][0]]]))
            else:
                input.append(np.array([s[feature]]))
        return input

    def load_network(self, file_name):
        """ used for each task
        not exists: choose the meta params for task to train start.
        others: choose the task params for task to train ongoing.
        """
        file_path_source = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                        '../', '../', '../', 'meta_round')
        file_path_target = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                        file_name + '.pt')
        if not os.path.exists(file_path_target):
            file_newest = sorted(
                os.listdir(file_path_source),
                key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            file_path_source = os.path.join(file_path_source, file_newest)
            shutil.copy(file_path_source, file_path_target)
            ckpt = torch.load(file_path_target)
        else:
            ckpt = torch.load(file_path_target)
        self.model = FRAPPlus(self.dic_traffic_env_conf)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lossfunc = torch.nn.MSELoss()
        self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def load_network_bar(self, file_name):
        """reference to the function `load_network` comment
        """
        if self.round_number == 1:
            self.model_target = FRAPPlus(self.dic_traffic_env_conf)
            self.model_target.load_state_dict(self.model.state_dict())
        else:
            file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                     file_name + '.pt')
            ckpt = torch.load(file_path)
            self.model_target = FRAPPlus(self.dic_traffic_env_conf)
            self.model_target.load_state_dict((ckpt['state_dict']))

    def load_network_meta(self):
        """used for meta agent
        """
        model_dir = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 'round_%d.pt' % (self.round_number - 1))
        ckpt = torch.load(model_dir)
        self.model_meta = FRAPPlus(self.dic_traffic_env_conf)
        self.model_meta.load_state_dict(ckpt['state_dict'])
        self.optimizer_meta = torch.optim.Adam(self.model_meta.parameters())
        self.optimizer_meta.load_state_dict(ckpt['optimizer'])
        self.lossfunc_meta = torch.nn.MSELoss()
        self.lossfunc_meta.load_state_dict(ckpt['lossfunc'])

    def prepare_Xs_Y(self, sample_set):
        """used for each task
        """
        state = []
        action = []
        next_state = []
        reward_avg = []
        for each in sample_set:
            state.append(each[0]['cur_phase'] + each[0]['lane_vehicle_cnt'])
            action.append(each[1])
            next_state.append(
                each[2]['cur_phase'] + each[2]['lane_vehicle_cnt'])
            reward_avg.append(each[3])

        q_values = self.model.forward(torch.Tensor(state)).detach().numpy()
        q_values_bar = self.model_target.forward(
            torch.Tensor(next_state)).detach().numpy()
        reward_avg = np.array(reward_avg) / self.dic_agent_conf["NORMAL_FACTOR"]
        gamma = self.dic_agent_conf["GAMMA"]
        range_idx = list(range(len(q_values)))
        try:
            q_values[range_idx, action] = \
                reward_avg + gamma * np.max(q_values_bar, axis=-1)
        except:
            print('a breakpoint here.')

        self.Xs = np.array(state)
        self.Y = q_values
        pass

    def prepare_Xs_Y_meta(self, sample_set):
        """used for each task <- note
        """
        state = []
        action = []
        next_state = []
        reward_avg = []
        for each in sample_set:
            state.append(each[0]['cur_phase'] + each[0]['lane_vehicle_cnt'])
            action.append(each[1])
            next_state.append(
                each[2]['cur_phase'] + each[2]['lane_vehicle_cnt'])
            reward_avg.append(each[3])

        q_values = self.model.forward(torch.Tensor(state)).detach().numpy()
        Xs = np.array(state)
        Y = q_values
        return Xs, Y

    def train_network(self):
        """used for each task
        """
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

    def train_network_meta(self, list_targets):
        """used for meta agent.
        """
        epochs = self.dic_agent_conf["EPOCHS"]
        idx = [i for i in range(len(list_targets))]

        for i in range(epochs):
            sample_x, sample_y = list_targets[idx, 0], list_targets[idx, 1]
            sample_x = np.array([each.tolist() for each in sample_x])
            sample_y = np.array([each.tolist() for each in sample_y])
            yp = self.model_meta.forward(torch.Tensor(sample_x))
            loss = self.lossfunc_meta(yp, torch.Tensor(sample_y))
            self.optimizer_meta.zero_grad()
            loss.backward()
            self.optimizer_meta.step()
            # print('%d memory, updating... %d, loss: %.4f'
            #       % (len(self.Y), i, loss.item()))

    def save_network(self, file_name):
        """
        """
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lossfunc': self.lossfunc.state_dict()}
        torch.save(ckpt, file_path)

    def save_network_meta(self, file_name):
        """
        """
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = {'state_dict': self.model_meta.state_dict(),
                'optimizer': self.optimizer_meta.state_dict(),
                'lossfunc': self.lossfunc_meta.state_dict()}
        torch.save(ckpt, file_path)
