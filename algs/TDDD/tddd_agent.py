import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from algs.agent import Agent


class Actor(nn.Module):
    def __init__(self, dic_traffic_env_conf, dic_agent_conf,
                 enable_context=False):
        super().__init__()
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.enable_context = enable_context

        self.lane_phase_info = self.dic_traffic_env_conf["LANE_PHASE_INFO"]

        phase_dim = dim_feature['cur_phase_index'][0]
        vehicle_dim = dim_feature['lane_vehicle_cnt'][0]
        self.state_dim = phase_dim + vehicle_dim
        self.action_dim = len(self.lane_phase_info['phase'])
        self.hidden_dim = self.dic_agent_conf["HIDDEN_DIM"]

        if self.enable_context:
            self.context = Context(self.dic_traffic_env_conf,
                                   self.dic_agent_conf)
        else:
            self.hidden_dim = 0

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim + self.hidden_dim, 100),
            nn.Sigmoid(),
            nn.Linear(100, 80),
            nn.Sigmoid(),
            nn.Linear(80, self.action_dim),
            nn.Sigmoid(),
        )

    def forward(self, state, pre_act_rew=None):
        if self.enable_context:
            hidden_msg = self.context(pre_act_rew)
            state = torch.cat([state, hidden_msg], dim=-1)
        action_prob = self.actor(state)
        return action_prob


class Critic(nn.Module):
    def __init__(self, dic_traffic_env_conf, dic_agent_conf,
                 enable_context=False):
        super(Critic, self).__init__()
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.enable_context = enable_context

        self.lane_phase_info = self.dic_traffic_env_conf["LANE_PHASE_INFO"]
        dim_feature = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]
        phase_dim = dim_feature['cur_phase_index'][0]
        vehicle_dim = dim_feature['lane_vehicle_cnt'][0]
        self.state_dim = phase_dim + vehicle_dim
        self.action_dim = len(self.lane_phase_info['phase'])
        self.input_dim = self.action_dim + 1 + self.state_dim

        if self.enable_context:
            self.context = Context(self.dic_traffic_env_conf,
                                   self.dic_agent_conf)
        else:
            self.input_dim = 0

        self.q1 = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + self.input_dim,
                      200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + self.input_dim,
                      200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, state, action_prob, pre_act_rew=None):

        state_action = torch.cat([state, action_prob], dim=-1)
        if self.enable_context:
            combined = self.context(pre_act_rew)
            state_action = torch.cat([state_action, combined], dim=-1)

        value1 = self.q1(state_action)
        value2 = self.q2(state_action)
        return value1, value2

    def Q1(self, state, action_prob, pre_act_rew=None):
        state_action = torch.cat([state, action_prob], 1)
        if self.enable_context:
            combined = self.context(pre_act_rew)
            state_action = torch.cat([state_action, combined], dim=-1)
        value1 = self.q1(state_action)
        return value1


class TDDDAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 round_number):
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path,
                         round_number)

        if self.round_number == 0:
            self.build_network()
            self.build_network_bar()
        else:
            self.load_network("round_%d" % (self.round_number - 1))
            self.load_network_bar("round_%d" % (self.round_number - 1))
        self.action_prob = []

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

    def choose_action(self, state, hidden=None):
        input = self.convert_state_to_input(state)
        input = torch.Tensor(input).flatten().unsqueeze(0)
        action_prob = self.model_actor.forward(input)[0].detach().numpy()
        action_prob = self.convert_action_prob(action_prob, mode="explore")
        action_prob = action_prob.squeeze()

        action = np.argmax(action_prob)
        self.action_prob.append(action_prob)
        return action

    def save_action_prob(self):
        work_path = self.dic_path["PATH_TO_WORK"]
        for each_file in os.listdir(work_path):
            if ".pkl" in each_file:
                file_name = os.path.join(work_path, each_file)
                with open(file_name, "rb") as f:
                    logging_data = pickle.load(f)
                length_cnt = 0
                for each_data in logging_data:
                    if each_data["action"] != -1:
                        each_data["action"] = self.action_prob[length_cnt]
                        length_cnt += 1
                if length_cnt != len(self.action_prob):
                    raise ValueError(length_cnt, " vs ", len(self.action_prob))
                with open(file_name, "wb") as f:
                    pickle.dump(logging_data, f)

    def convert_action_prob(self, action_prob, mode):
        if mode == "explore":
            sigma = self.dic_agent_conf["EXPL_NOISE"]
            clip_min = self.dic_agent_conf["EXPL_NOISE_MIN"]
            clip_max = self.dic_agent_conf["EXPL_NOISE_MAX"]
            explore_noise = np.random.normal(0, sigma, size=action_prob.shape)
            explore_noise = np.clip(explore_noise, clip_min, clip_max)
            action_prob += explore_noise
            return action_prob
        elif mode == "policy":
            idx = list(range(len(action_prob)))
            action_prob[idx, np.argmax(action_prob, axis=-1)] = 1
            action_prob[action_prob != 1] = 0
            return action_prob
        elif mode == "replay":
            sigma = 0
            clip_min = 0
            clip_max = 0
            raise NotImplementedError(mode)
        else:
            raise ValueError(mode)

    def build_network(self):
        self.model_actor = Actor(self.dic_traffic_env_conf,
                                 self.dic_agent_conf)
        self.model_critic = Critic(self.dic_traffic_env_conf,
                                   self.dic_agent_conf)
        self.loss_func_actor = torch.nn.MSELoss()
        self.loss_func_critic = torch.nn.MSELoss()
        self.optimizer_actor = \
            torch.optim.Adam(self.model_actor.parameters(),
                             lr=self.dic_agent_conf["LR_ACTOR"])
        self.optimizer_critic = torch.optim.Adam(
            self.model_critic.parameters(), lr=self.dic_agent_conf["LR"])

    def build_network_bar(self):
        self.model_actor_bar = Actor(self.dic_traffic_env_conf,
                                     self.dic_agent_conf)
        self.model_critic_bar = Critic(self.dic_traffic_env_conf,
                                       self.dic_agent_conf)
        self.model_actor_bar.load_state_dict(self.model_actor_bar.state_dict())
        self.model_critic_bar.load_state_dict(self.model_critic.state_dict())
        pass

    def load_network(self, file_name):
        self.build_network()

        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_actor.pt')
        ckpt = torch.load(file_path)
        self.model_actor.load_state_dict(ckpt["state_dict"])
        self.optimizer_actor.load_state_dict(ckpt["optimizer"])
        self.loss_func_actor.load_state_dict(ckpt["lossfunc"])

        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_critic.pt')
        ckpt = torch.load(file_path)
        self.model_critic.load_state_dict(ckpt["state_dict"])
        self.optimizer_critic.load_state_dict(ckpt["optimizer"])
        self.loss_func_critic.load_state_dict(ckpt["lossfunc"])
        pass

    def load_network_bar(self, file_name):
        self.model_actor_bar = Actor(self.dic_traffic_env_conf,
                                     self.dic_agent_conf)
        self.model_critic_bar = Critic(self.dic_traffic_env_conf,
                                       self.dic_agent_conf)

        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_actor_bar.pt')
        ckpt = torch.load(file_path)
        self.model_actor_bar.load_state_dict(ckpt["state_dict"])

        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_critic_bar.pt')
        ckpt = torch.load(file_path)
        self.model_critic_bar.load_state_dict(ckpt["state_dict"])
        pass

    def prepare_Xs_Y(self, sample_set):
        state = []
        action_prob = []
        next_state = []
        reward_avg = []
        for each in sample_set:
            state.append(each[0]['cur_phase_index'] + each[0]['lane_vehicle_cnt'])
            action_prob.append(each[1])
            next_state.append(
                each[2]['cur_phase_index'] + each[2]['lane_vehicle_cnt'])
            reward_avg.append(each[3])

        action_prob_bar = self.model_actor_bar.forward(
            torch.Tensor(state)).detach().numpy()
        action_prob_bar = \
            self.convert_action_prob(action_prob_bar, mode="policy")

        q_values_bar1, q_values_bar2 = self.model_critic_bar.forward(
            torch.Tensor(next_state), torch.Tensor(action_prob_bar))
        q_values_bar = torch.min(q_values_bar1, q_values_bar2).detach().numpy()

        reward_avg = np.array(reward_avg) / self.dic_agent_conf["NORMAL_FACTOR"]
        reward_avg = reward_avg.reshape(q_values_bar.shape[0], -1)
        gamma = self.dic_agent_conf["GAMMA"]

        value_y = reward_avg + gamma * q_values_bar

        self.Xs = np.array(state), np.array(action_prob)
        self.Y = value_y
        pass

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        for i in range(epochs):
            state, action_prob = self.Xs
            sample_y = self.Y
            yp1, yp2 = self.model_critic.forward(torch.Tensor(state),
                                                 torch.Tensor(action_prob))
            self.optimizer_critic.zero_grad()
            loss_critic = self.loss_func_critic(yp1, torch.Tensor(sample_y)) + \
                          self.loss_func_critic(yp2, torch.Tensor(sample_y))
            loss_critic.backward()
            self.optimizer_critic.step()

            if self.round_number % self.dic_agent_conf["POLICY_FREQ"] == 0:
                action_prob = self.model_actor.forward(torch.Tensor(state))
                q_value = self.model_critic.Q1(torch.Tensor(state), action_prob)
                self.optimizer_actor.zero_grad()
                loss_actor = -torch.mean(q_value)
                loss_actor.backward()
                self.optimizer_actor.step()

                print('updating... %d, loss_actor: %.4f, loss_critic:%.4f'
                      % (i, loss_actor.item(), loss_critic.item()))
                # print("yp1 %s, yp2 %s" % (torch.sum(yp1), torch.sum(yp2)))

        if self.round_number % self.dic_agent_conf["POLICY_FREQ"] == 0:
            self.update_model_bar()
        pass

    def update_model_bar(self):
        tau = self.dic_agent_conf["TAU"]
        for t, s in zip(self.model_critic_bar.parameters(),
                        self.model_critic.parameters()):
            t.data.copy_(s.data * (1.0 - tau) + s.data * tau)
        for t, s in zip(self.model_actor_bar.parameters(),
                        self.model_actor.parameters()):
            t.data.copy_(s.data * (1.0 - tau) + s.data * tau)

    def save_network(self, file_name):
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_actor.pt')
        ckpt = {'state_dict': self.model_actor.state_dict(),
                'optimizer': self.optimizer_actor.state_dict(),
                'lossfunc': self.loss_func_actor.state_dict()}
        torch.save(ckpt, file_path)

        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_critic.pt')
        ckpt = {'state_dict': self.model_critic.state_dict(),
                'optimizer': self.optimizer_critic.state_dict(),
                'lossfunc': self.loss_func_critic.state_dict()}
        torch.save(ckpt, file_path)
        # ---------------------save network bar--------------------------------
        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_actor_bar.pt')
        ckpt = {'state_dict': self.model_actor_bar.state_dict()}
        torch.save(ckpt, file_path)

        file_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                 file_name + '_critic_bar.pt')
        ckpt = {'state_dict': self.model_critic_bar.state_dict()}
        torch.save(ckpt, file_path)
