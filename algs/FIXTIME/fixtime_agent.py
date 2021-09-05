class FIXTIMEAgent():
    def __init__(self, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

        self.lane_phase_info = dic_traffic_env_conf["LANE_PHASE_INFO"]
        self.phase_cnt = len(self.lane_phase_info['phase'])

        self.list_action_time = []
        self.list_action = []
        for i in range(self.phase_cnt):
            self.list_action_time += [self.dic_agent_conf["TIME_PHASE_%d" % i]]
        for phase, each in enumerate(self.list_action_time):
            self.list_action += [phase for _ in range(int(round(each)))]

        self.global_cnt = 0

    def choose_action(self, state):
        action = self.list_action[self.global_cnt % len(self.list_action)]
        self.global_cnt += 1
        return action
