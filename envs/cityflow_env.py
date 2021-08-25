import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
from misc import utils
import json
from misc.utils import get_vehicle_list

"""
    Class CityFlowEnv provides the environment for traffic signal control
    of single (or multiple) intersections
    Class Intersection specifies the environment for single intersection
"""


class Intersection:
    def __init__(self, dic_traffic_env_conf, eng):
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.eng = eng

        self.inter_name = self.dic_traffic_env_conf["INTER_NAME"]
        self.num_phases = \
            len(self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"])
        self.list_entering_lanes = \
            dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
        self.list_exiting_lanes = \
            dic_traffic_env_conf["LANE_PHASE_INFO"]["end_lane"]
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes
        self.yellow_phase_index = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]["yellow_phase"]

        # previous & current
        self.current_phase_index = self.yellow_phase_index
        self.previous_phase_index = self.yellow_phase_index
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_yellow_flag = False

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative
        self.dic_feature = {}

    def set_signal(self, action, yellow_time):
        """Initial all_yellow_flag with False.
        First call, one action index is in , become yellow phase.
        Second to the yellow time call, keep yellow phase
        Third to the last call, because of the action list is same, keep action
        """
        if self.all_yellow_flag:
            if self.current_phase_duration >= yellow_time:
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
                self.all_yellow_flag = False
        else:
            self.next_phase_to_set_index = action
            if self.current_phase_index != self.next_phase_to_set_index:
                self.eng.set_tl_phase(self.inter_name, self.yellow_phase_index)
                self.current_phase_index = self.yellow_phase_index
                self.all_yellow_flag = True

    def update_previous_measurements(self):
        # same to __init__ function values.
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = \
            self.dic_lane_vehicle_current_step
        self.dic_lane_waiting_vehicle_count_previous_step = \
            self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = \
            self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = \
            self.dic_vehicle_distance_current_step

    def update_current_measurements(self):
        # same to __init__ function values. AND UPDATE FEATURE etc.
        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1
        # Update data from engine
        self.dic_lane_vehicle_current_step = self.eng.get_lane_vehicles()
        self.dic_lane_waiting_vehicle_count_current_step = \
            self.eng.get_lane_waiting_vehicle_count()
        if self.dic_traffic_env_conf["DEBUG"]:
            self.dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
            self.dic_vehicle_distance_current_step = \
                self.eng.get_vehicle_distance()
        # log vehicle id and its arrive and left time. get vehicle list
        vehicle_now = get_vehicle_list(self.dic_lane_vehicle_current_step)
        vehicle_pre = get_vehicle_list(self.dic_lane_vehicle_previous_step)
        list_vehicle_new_arrive = set(vehicle_now) - set(vehicle_pre)
        list_vehicle_new_left = set(vehicle_pre) - set(vehicle_now)
        # update vehicle arrive and left time. update feature
        self._update_enter_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
        self._update_feature()

    def _update_feature(self):
        dic_feature = dict()
        phase = [0] * len(self.list_entering_lanes)
        if self.current_phase_index != self.yellow_phase_index:
            start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"][
                "phase_startLane_mapping"][self.current_phase_index]
            for lane in start_lane:
                phase[self.list_entering_lanes.index(lane)] = 1

        dic_feature["cur_phase"] = phase
        dic_feature["cur_phase_index"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        # self._get_lane_vehicle_position(self.list_entering_lanes)
        # self._get_lane_vehicle_speed(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = \
            [len(self.dic_lane_vehicle_current_step[lane])
             for lane in self.list_entering_lanes]
        dic_feature["lane_num_vehicle_been_stopped_thres01"] = \
            self._get_lane_num_vehicle_been_stopped(0.1,
                                                    self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = \
            self._get_lane_num_vehicle_been_stopped(1,
                                                    self.list_entering_lanes)
        dic_feature["lane_queue_length"] = \
            self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = None
        dic_feature["terminal"] = None

        self.dic_feature = dic_feature

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append(
                    list(
                        set(self.dic_lane_vehicle_previous_step[lane]) -
                        set(self.dic_lane_vehicle_current_step[lane])
                    )
                )
        return list_entering_lane_vehicle_left

    def _update_enter_time(self, vehicle_arrive):
        ts = self.eng.get_current_time()
        for vehicle in vehicle_arrive:
            self.dic_vehicle_arrive_leave_time[vehicle] = \
                {"enter_time": ts, "leave_time": np.nan}

    def _update_left_time(self, vehicle_left):
        ts = self.eng.get_current_time()
        for vehicle in vehicle_left:
            self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts

    # ================= calculate features from current observations =========
    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_waiting_vehicle_count_current_step[lane]
                for lane in list_lanes]

    def _get_lane_num_vehicle_left(self, list_lanes):
        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [
            len(lane_vehicle_left) for lane_vehicle_left in
            list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        if self.dic_traffic_env_conf['INPUT_NORM']:
            return [
                self.dic_lane_waiting_vehicle_count_current_step[lane] /
                40 for lane in list_lanes]
        else:
            return [self.dic_lane_waiting_vehicle_count_current_step[lane]
                    for lane in list_lanes]

    def _get_lane_vehicle_position(self, list_lanes):
        self.length_grid = 5
        self.num_grid = int(300 // 50)

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_vehicle_current_step[lane]
            for vec in list_vec_id:
                pos = int(self.dic_vehicle_distance_current_step[vec])
                pos_grid = min(pos // self.length_grid, self.num_grid)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    # debug
    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_distance_current_step[veh_id]
            speed = self.dic_vehicle_speed_current_step[veh_id]
            return pos, speed
        except BaseException:
            return None, None

    def _get_lane_vehicle_speed(self, list_lanes):
        return [self.dic_vehicle_speed_current_step[lane]
                for lane in list_lanes]

    # ================= get functions from outside ======================

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {k: self.dic_feature[k] for k in list_state_features}
        return dic_state

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(
            self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        if self.dic_traffic_env_conf['REWARD_NORM']:
            # normalize the reward
            reward = - 2 * dic_reward["sum_num_vehicle_been_stopped_thres1"] / (
                    40 * len(
                self.dic_feature["lane_num_vehicle_been_stopped_thres1"])) + 1
        else:
            reward = 0
            for r in dic_reward_info:
                if dic_reward_info[r] != 0:
                    reward += dic_reward_info[r] * dic_reward[r]
        return reward


class CityFlowEnv:
    def __init__(self, path_to_log, path_to_work_directory,
                 dic_traffic_env_conf):
        """Base environment

        Args:
            path_to_log:
            path_to_work_directory:
            dic_traffic_env_conf:
        """
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self._create_eng()
        self.stop_cnt = 0

    def _create_eng(self):
        """
            The following commented codes shows the latest API of cityflow.
            However, it will somehow hang in the multi-process.
            Therefore, another version of cityflow is used here,
            "engine.cpython-36m-x86_64-linux-gnu.so meta2:/metalight"
        Returns:
            self.eng from cityflow or engine which is customized by others.
        """
        vol = utils.get_total_traffic_volume(
            self.dic_traffic_env_conf["TRAFFIC_FILE"])
        if self.dic_traffic_env_conf["USE_CITYFLOW"]:
            warn("using cityflow may stuck in next_step.")
            import cityflow as engine
            config_dict = {
                "interval": self.dic_traffic_env_conf["INTERVAL"],
                "seed": 0,
                "dir": "",
                "roadnetFile":
                    os.path.join(self.path_to_work_directory,
                                 self.dic_traffic_env_conf['ROADNET_FILE']),
                "flowFile":
                    os.path.join(self.path_to_work_directory,
                                 self.dic_traffic_env_conf["FLOW_FILE"]),
                "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
                "roadnetLogFile": os.path.join(self.path_to_log,
                                               "roadnet_%d.json" % vol),
                "replayLogFile": os.path.join(self.path_to_log,
                                              "replay_%d.txt" % vol),
            }
            config_path = os.path.join(self.path_to_log, "cityflow_config")
            with open(config_path, "w") as f:
                json.dump(config_dict, f)
                print("dump cityflow config")
                print(config_path)
            self.eng = engine.Engine(
                config_path, self.dic_traffic_env_conf["THREADNUM"])
        else:
            import engine
            self.eng = engine.Engine(
                self.dic_traffic_env_conf["INTERVAL"],
                self.dic_traffic_env_conf["THREADNUM"],
                self.dic_traffic_env_conf["SAVEREPLAY"],
                self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                False)

            self.eng.load_roadnet(
                os.path.join(self.path_to_work_directory,
                             self.dic_traffic_env_conf["ROADNET_FILE"]))
            self.eng.load_flow(
                os.path.join(self.path_to_work_directory,
                             self.dic_traffic_env_conf["FLOW_FILE"]))
            print("successfully load files.")

    def reset(self):
        self.eng.reset()

        self.list_intersection = \
            [Intersection(self.dic_traffic_env_conf, self.eng)]

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

        self.list_inter_log = [[] for _ in self.list_intersection]
        self.reward_total = [0 for i in self.list_intersection]

        state = self.get_state()

        return state

    def step(self, action):
        # action extends with MIN_ACTION_TIME for step several
        list_action_in_sec = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"] - 1):
            list_action_in_sec.append(np.copy(action).tolist())

        average_reward = 0
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            current_time = self.eng.get_current_time()

            state = self.get_feature()
            action = list_action_in_sec[i]
            self._inner_step(action)
            rewards = self.get_rewards()
            average_reward = (average_reward * i + sum(rewards)) / (i + 1)
            next_state = self.get_state()
            done = self._check_episode_done(next_state)

            self.inter_log(current_time, state, action,
                           next_state, rewards, done)

        print('.', end='')
        print(action, end='')
        if done:
            print('||done||')

        for i in range(len(self.list_intersection)):
            self.reward_total[i] += average_reward

        return next_state, rewards, done, average_reward

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        for i, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[i],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"])

        # run one step
        for i in range(int(1 / self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()
        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

    def _check_episode_done(self, state):
        if self.eng.get_current_time() >= \
                self.dic_traffic_env_conf['EPISODE_LEN']:
            return True
        else:
            if self.dic_traffic_env_conf["DONE_ENABLE"]:
                if 39 in state[0]["lane_num_vehicle"]:
                    self.stop_cnt += 1

                if self.stop_cnt == 100:
                    self.stop_cnt = 0
                    return True
                else:
                    return False
            else:
                return False

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def get_feature(self):
        list_feature = [inter.get_feature()
                        for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        list_state = \
            [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
             for inter in self.list_intersection]

        return list_state

    def get_rewards(self):
        list_reward = \
            [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"])
             for inter in self.list_intersection]

        return list_reward

    def inter_log(self, current_time, state, action, next_state, rewards, done):
        """Store intersection trace in self.list_inter_log
        """
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append(
                {"current_time": current_time,
                 "state": state[inter_ind],
                 "action": action[inter_ind],
                 "next_state": next_state[inter_ind],
                 "reward": rewards[inter_ind],
                 "done": done})

    def bulk_log(self):
        record_msg = {}

        for i in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(
                self.path_to_log, "vehicle_inter_{0}.csv".format(i))
            dic_vehicle = self.list_intersection[
                i].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

            inter = self.list_intersection[i]
            feature = inter.get_feature()

            if max(feature['lane_num_vehicle']) > \
                    self.dic_traffic_env_conf["VALID_THRESHOLD"]:
                record_msg["inter_valid_" + str(i)] = 0
            else:
                record_msg["inter_valid_" + str(i)] = 1
            record_msg["inter_reward_" + str(i)] = self.reward_total[i]

        json.dump(record_msg,
                  open(os.path.join(self.path_to_log, "record_msg.json"), "w"))

        if not self.dic_traffic_env_conf["USE_CITYFLOW"]:
            self.log_replay()

    def log_replay(self):
        vol = utils.get_total_traffic_volume(
            self.dic_traffic_env_conf["TRAFFIC_FILE"])
        self.eng.print_log(
            os.path.join(self.path_to_log, "roadnet_%s.json" % vol),
            os.path.join(self.path_to_log, "replay_%s.txt" % vol))
