import numpy as np

from algs.agent_fix import AgentFix


class MAXPRESSUREAgent(AgentFix):
    def __init__(self, conf_path, round_number, inter_name):
        super().__init__(conf_path, round_number, inter_name)

        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__inter_name = inter_name
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file(inter_name=inter_name)

        self.lane_phase_info = self.__conf_traffic.TRAFFIC_INFO
        self.phase_count = len(self.lane_phase_info["phase_lane_mapping"].keys())
        self.phase_map = self.lane_phase_info["phase_lane_mapping"]

        self.g_min = self.__conf_agent["G_MIN"]

        self.global_cnt = 0
        self.list_action = []

    def get_phase_method(self, state):
        x = np.array(state["lane_vehicle_cnt"])
        y = np.array(state["lane_vehicle_left_cnt"])
        lane_pressure = x - y

        phase_pressure = []
        for phase in self.phase_map:
            location = np.array(self.phase_map[phase]) == 1
            phase_pressure += [sum(lane_pressure[location])]
        action = np.argmax(phase_pressure)
        self.list_action = [action for _ in range(self.g_min)]

    def choose_action(self, state, choice_random=False):
        if len(self.list_action) == 0:
            self.get_phase_method(state)
        action = self.list_action[0]
        del self.list_action[0]
        return action
