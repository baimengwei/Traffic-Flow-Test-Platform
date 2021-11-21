from algs.agent_fix import AgentFix


class SOTLAgent(AgentFix):
    def __init__(self, conf_path, round_number, inter_name):
        super().__init__(conf_path, round_number, inter_name)
        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__inter_name = inter_name
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file(inter_name=inter_name)

        self.lane_phase_info = self.__conf_traffic.TRAFFIC_INFO

        self.current_phase_time = 0
        self.kappa = 0
        self.global_cnt = 0
        self.phase_count = len(self.lane_phase_info["phase_lane_mapping"].keys())
        self.phase_map = self.lane_phase_info["phase_lane_mapping"]

    def choose_action(self, state, choice_random=False):
        cur_phase = state['cur_phase_index']
        lane_stop_vehicle = state['stop_vehicle_thres1']
        self.kappa += sum(lane_stop_vehicle)

        action = cur_phase
        self.global_cnt += 1

        if self.current_phase_time > self.__conf_agent["PHI_MIN"]:
            green_vehicle = \
                sum([lane_stop_vehicle[idx]
                     for idx, lane_enable in enumerate(self.phase_map[action - 1])
                     if lane_enable == 1])
            if not 0 < green_vehicle < self.__conf_agent["MU"]:
                if self.kappa > self.__conf_agent["THETA"]:
                    action = (cur_phase) % self.phase_count
                    self.kappa = 0
                    self.current_phase_time = 0
                    return action
        else:
            self.current_phase_time += 1
        # print('time %3d action: %d' % (self.global_cnt, action))
        return action - 1
