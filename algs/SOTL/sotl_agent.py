class SOTLAgent():
    def __init__(self, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

        self.current_phase_time = 0
        self.kappa = 0
        self.global_cnt = 0
        self.lane_phase_info = self.dic_traffic_env_conf["LANE_PHASE_INFO"]
        self.phase_count = len(self.lane_phase_info["phase"])
        self.phase_map = self.lane_phase_info["phase_map"]

    def choose_action(self, state):
        cur_phase = state['cur_phase'][0]
        lane_stop_vehicle = state['stop_vehicle_thres1']
        self.kappa += sum(lane_stop_vehicle)

        action = cur_phase
        self.global_cnt += 1

        if self.current_phase_time > self.dic_agent_conf["PHI_MIN"]:
            green_vehicle = \
                sum([lane_stop_vehicle[idx]
                     for idx, lane_enable in enumerate(self.phase_map[action])
                     if lane_enable == 1])
            if not 0 < green_vehicle < self.dic_agent_conf["MU"]:
                if self.kappa > self.dic_agent_conf["THETA"]:
                    action = (cur_phase) % self.phase_count
                    self.kappa = 0
                    self.current_phase_time = 0
                    return action
        else:
            self.current_phase_time += 1
        # print('time %3d action: %d' % (self.global_cnt, action))
        return action-1
