import numpy as np

from algs.agent_fix import AgentFix


class WEBSTERAgent(AgentFix):
    def __init__(self, conf_path, round_number, inter_name):
        super().__init__(conf_path, round_number, inter_name)
        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__inter_name = inter_name
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file(inter_name=inter_name)

        self.traffic_info = self.__conf_traffic.TRAFFIC_INFO


        self.vehicle_dim = len(self.traffic_info['phase_links'])
        self.R = self.__conf_agent["L_LANE"] * self.vehicle_dim
        self.flow_rate = [0 for _ in range(self.vehicle_dim)]
        self.flow_cnt = [0 for _ in range(self.vehicle_dim)]
        self.lane_num_vehicle_pre = [0 for _ in range(self.vehicle_dim)]

        self.phase_cnt = len(self.traffic_info['phase_lane_mapping'])
        self.dic_phase_expansion = self.traffic_info['phase_lane_mapping']
        self.global_cnt = 0
        self.global_cnt_pre = 0
        self.circle = 30
        self.get_phase_method()

    def get_value_Y(self):
        ymax = self.__conf_agent["Y_MAX"]
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
        k1 = self.__conf_agent["K1"]
        k2 = self.__conf_agent["K2"]
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

    def choose_action(self, state, choice_random=False):
        x = self.lane_num_vehicle_pre - np.array(state["lane_vehicle_cnt"])
        for idx, each in enumerate(x):
            if each > 0:
                self.flow_cnt[idx] += 1
        self.lane_num_vehicle_pre = np.array(state["lane_vehicle_cnt"])

        if len(self.list_action) > 0:
            self.global_cnt += 1
            action = self.list_action[0]
            del self.list_action[0]
            return action
        else:
            self.get_phase_method()
            return self.choose_action(state)
