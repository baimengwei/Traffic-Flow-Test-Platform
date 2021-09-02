import numpy as np
import pickle
import os
from misc.utils import set_seed

class ConstructSample:

    def __init__(self, path_to_samples, round_number, dic_traffic_env_conf):

        self.path_to_samples = path_to_samples
        self.round_number = round_number
        self.dic_traffic_env_conf = dic_traffic_env_conf

        set_seed(self.round_number)

        self.parent_dir = os.path.join(self.path_to_samples, '../')

    def load_data(self, folder):
        self.measure_time = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        for inter_name in self.dic_traffic_env_conf["LANE_PHASE_INFOS"].keys():
            file_name = os.path.join(self.path_to_samples, folder,
                                     "%s.pkl" % inter_name)
            f_logging_data = open(file_name, "rb")
            self.logging_data = pickle.load(f_logging_data)
            f_logging_data.close()

    def construct_state(self, features, time):
        state = self.logging_data[time]
        assert time == state["time"]
        phase_expansion = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]['phase_map']
        state_after_selection = {}
        for key, value in state["state"].items():
            if key in features:
                if key == "cur_phase":
                    state_after_selection[key] = phase_expansion[value[0]]
                else:
                    state_after_selection[key] = value
        return state_after_selection

    def get_reward_from_features(self, rs):
        reward = dict()
        reward["sum_lane_queue_length"] = np.sum(rs["lane_queue_length"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_sum_waiting_time"])
        reward["sum_lane_num_vehicle_left"] = np.sum(
            rs["lane_num_vehicle_left"])
        reward["sum_duration_vehicle_left"] = np.sum(
            rs["lane_sum_duration_vehicle_left"])
        reward["sum_stop_vehicle_thres01"] = np.sum(
            rs["stop_vehicle_thres01"])
        reward["sum_stop_vehicle_thres1"] = np.sum(
            rs["stop_vehicle_thres1"])
        return reward

    def cal_reward(self, rs, rewards_components):
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        return r

    def construct_reward(self, rewards_components, time):
        rs = self.logging_data[time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])
        r_instant = self.cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):
            # print("t is ", t)
            rs = self.logging_data[t]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    def judge_action(self, time):
        if self.logging_data[time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data[time]['action']

    def make_reward(self):
        self.samples = []
        for folder in os.listdir(self.path_to_samples):
            if "generator" not in folder:
                continue

            self.load_data(folder)

            list_samples = []
            total_time = int(self.logging_data[-1]['time'] + 1)
            # construct samples
            for time in range(0, total_time - self.measure_time + 1,
                              self.interval):
                state = self.construct_state(
                    self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time)
                reward_instant, reward_average = self.construct_reward(
                    self.dic_traffic_env_conf["DIC_REWARD_INFO"], time)
                action = self.judge_action(time)

                if time + self.interval == total_time:
                    next_state = self.construct_state(
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
                        , time + self.interval - 1)
                else:
                    next_state = self.construct_state(
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                        time + self.interval)
                sample = [state, action, next_state, reward_average,
                          reward_instant, time]
                list_samples.append(sample)

            self.samples.extend(list_samples)

        self.dump_sample(self.samples, "")

    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"),
                      "ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples,
                                   folder, "samples_{0}.pkl".format(folder)),
                      'wb') as f:
                pickle.dump(samples, f, -1)
