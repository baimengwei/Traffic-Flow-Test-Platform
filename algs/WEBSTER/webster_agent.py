import numpy as np


class WEBSTERAgent():
    def __init__(self, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

        self.lane_phase_info = dic_traffic_env_conf["LANE_PHASE_INFO"]
        dim_feature = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]

        self.vehicle_dim = dim_feature['lane_num_vehicle'][0]
        self.R = self.dic_agent_conf["L_LANE"] * self.vehicle_dim
        self.flow_rate = [0 for _ in range(self.vehicle_dim)]
        self.flow_cnt = [0 for _ in range(self.vehicle_dim)]
        self.lane_num_vehicle_pre = [0 for _ in range(self.vehicle_dim)]

        self.phase_cnt = len(self.lane_phase_info['phase'])
        self.dic_phase_expansion = self.lane_phase_info['phase_map']
        self.global_cnt = 0
        self.global_cnt_pre = 0
        self.circle = 30
        self.get_phase_method()

    def get_value_Y(self):
        ymax = self.dic_agent_conf["Y_MAX"]
        self.Y = []
        self.flow_rate = np.array(self.flow_cnt) * 3600 / (
                self.global_cnt - self.global_cnt_pre + 1e-8)
        self.global_cnt_pre = self.global_cnt
        self.flow_cnt = [0 for _ in range(self.vehicle_dim)]

        for i in range(0, len(self.flow_rate), 2):
            self.Y += [self.flow_rate[i] / ymax + 1e-8]
            self.Y += [self.flow_rate[i + 1] / ymax + 1e-8]
        self.Y = np.array(self.Y) / len(self.Y)

    def get_value_circle(self):
        k1 = self.dic_agent_conf["K1"]
        k2 = self.dic_agent_conf["K2"]
        self.circle = (self.R * k1 + k2) / (1 - sum(self.Y))

    def get_phase_method(self):
        self.get_value_Y()
        self.get_value_circle()
        self.list_action_time = \
            self.circle * ((np.array(self.Y)) / sum(self.Y))
        self.list_action_phase_time = []
        for each in self.dic_phase_expansion:
            location = np.array(self.dic_phase_expansion[each]) == 1
            self.list_action_phase_time += \
                [sum(self.list_action_time[location])]
        self.list_action = []
        for phase, each in enumerate(self.list_action_phase_time):
            self.list_action += [phase for _ in range(int(round(each)))]
        if self.dic_traffic_env_conf["ENV_DEBUG"]:
            print("\n step %d, phase method: %s" %
                  (self.global_cnt, self.list_action))

    def choose_action(self, state):
        x = self.lane_num_vehicle_pre - np.array(state["lane_num_vehicle"])
        for idx, each in enumerate(x):
            if each > 0:
                self.flow_cnt[idx] += 1
        self.lane_num_vehicle_pre = np.array(state["lane_num_vehicle"])

        if len(self.list_action) > 0:
            self.global_cnt += 1
            action = self.list_action[0]
            del self.list_action[0]
            return action
        else:
            self.get_phase_method()
            return self.choose_action(state)
