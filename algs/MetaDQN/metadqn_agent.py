import os.path
from algs.DQN.dqn_agent import DQN
from algs.agent import Agent
from configs.config_phaser import *
import torch
import numpy as np
import random


class MetaDQNAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 round_number, traffic_tasks):
        """mode is mata or task, which is supposed to define. default is 'task'
        """
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path,
                         round_number)
        self.traffic_tasks = traffic_tasks

        self.decay_epsilon(self.round_number)
        if self.round_number == 0:
            self.build_network()
        else:
            self.load_network("round_%d" % (self.round_number - 1))
            self.dic_model_tasks = {}
            self.load_network_bar(self.traffic_tasks)

    def build_network(self):
        """used for meta agent.
        """
        self.model = DQN(self.dic_traffic_env_conf)
        self.lossfunc = torch.nn.MSELoss()
        self.optimizer = \
            torch.optim.Adam(self.model.parameters(),
                             lr=self.dic_agent_conf["LR"])

    def build_network_bar(self):
        print('use pretrained.')
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
                input.append(np.array([dic_phase_expansion[s[feature]]]))
            else:
                input.append(np.array([s[feature]]))
        return input

    def load_network(self, file_name):
        """ used for meta
        """
        file_path_meta = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                      file_name + '.pt')

        ckpt = torch.load(file_path_meta)
        self.model = DQN(self.dic_traffic_env_conf)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lossfunc = torch.nn.MSELoss()
        self.lossfunc.load_state_dict(ckpt['lossfunc'])

    def load_network_bar(self, traffic_tasks):
        """reference to the function `load_network` comment
        """
        folder_path_task = os.path.join(
            self.dic_path["PATH_TO_MODEL"], '../', 'tasks_param')
        list_folder_name = os.listdir(folder_path_task)

        for folder in list_folder_name:
            for task in traffic_tasks:
                if task in folder:
                    params_dir = os.path.join(folder_path_task, folder)
                    params_files = os.listdir(params_dir)
                    params_files = \
                        sorted(params_files, key=lambda x: int(
                            x.split('.pt')[0].split('_')[-1]))
                    params_file = os.path.join(params_dir, params_files[-1])

                    ckpt = torch.load(params_file)
                    model = DQN(self.dic_traffic_env_conf)
                    model.load_state_dict(ckpt['state_dict'])
                    self.dic_model_tasks[task] = model

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
        # load target net.
        task = self.dic_traffic_env_conf["TRAFFIC_FILE"]
        model_target = self.dic_model_tasks[task]

        q_values = self.model.forward(torch.Tensor(state)).detach().numpy()
        q_values_bar = model_target.forward(
            torch.Tensor(next_state)).detach().numpy()
        reward_avg = np.array(reward_avg) / self.dic_agent_conf["NORMAL_FACTOR"]
        gamma = self.dic_agent_conf["GAMMA"]
        range_idx = list(range(len(q_values)))

        q_values[range_idx, action] = \
            reward_avg + gamma * np.max(q_values_bar, axis=-1)

        self.Xs = np.array(state)
        self.Y = q_values
        pass

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

    def save_network(self, file_name):
        """
        """
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '.pt')
        ckpt = {'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lossfunc': self.lossfunc.state_dict()}
        torch.save(ckpt, file_path)
