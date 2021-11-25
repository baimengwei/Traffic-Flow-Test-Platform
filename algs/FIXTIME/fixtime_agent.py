import numpy as np

from algs.agent_fix import AgentFix


class FIXTIMEAgent(AgentFix):
    def __init__(self, conf_path, round_number, inter_name):
        super().__init__(conf_path, round_number, inter_name)

        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__inter_name = inter_name
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file(inter_name=inter_name)

        self.list_action_time = []
        self.list_action = []
        phase_lane = self.__conf_traffic.TRAFFIC_INFO['phase_lane_mapping']
        self.phase_cnt = len(phase_lane)
        for i in range(self.phase_cnt):
            self.list_action_time += [self.__conf_agent["TIME_PHASE"]]
        for phase, each in enumerate(self.list_action_time):
            self.list_action += [phase for _ in range(int(round(each)))]
        self.global_cnt = 0

    def choose_action(self, state, choice_random=False):
        # if choice_random is True:
        #     return np.random.choice(self.list_action,1)[0]
        action = self.list_action[self.global_cnt % len(self.list_action)]
        self.global_cnt += 1
        return action
