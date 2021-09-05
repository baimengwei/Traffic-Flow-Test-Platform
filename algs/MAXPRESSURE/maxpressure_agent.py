import numpy as np


class MAXPRESSUREAgent:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

        self.lane_phase_info = self.dic_traffic_env_conf["LANE_PHASE_INFO"]
        self.phase_count = len(self.lane_phase_info["phase"])
        self.phase_map = self.lane_phase_info["phase_map"]

        self.g_min = self.dic_agent_conf["G_MIN"]

        self.global_cnt = 0
        self.list_action = []

    def get_phase_method(self, state):
        x = np.array(state["lane_num_vehicle"])
        y = np.array(state["lane_num_vehicle_left"])
        lane_pressure = x - y
        if self.dic_traffic_env_conf["ENV_DEBUG"]:
            print("\n x: %s, y: %s"%(x, y))
        phase_pressure = []
        for phase in self.phase_map:
            location = np.array(self.phase_map[phase]) == 1
            phase_pressure += [sum(lane_pressure[location])]
        action = np.argmax(phase_pressure)
        self.list_action = [action for _ in range(self.phase_count)]

    def choose_action(self, state):
        if len(self.list_action) == 0:
            self.get_phase_method(state)
        action = self.list_action[0]
        del self.list_action[0]
        return action
