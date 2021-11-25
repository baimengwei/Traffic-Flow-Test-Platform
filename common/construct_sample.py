import pickle

import numpy as np

from configs.config_phaser import *


class ConstructSample:
    def __init__(self, conf_path: ConfPath, round_number):
        self.__conf_path = conf_path
        self.__round_number = round_number

        list_inters = sorted(list(self.__conf_path.load_conf_inters()))

        self.__conf_exp, _, self.conf_traffic = \
            self.__conf_path.load_conf_file(inter_name=list_inters[0])

        self.__conf_path.set_work_sample_each(
            self.__round_number, self.__conf_exp.NUM_GENERATORS, list_inters)
        self.__conf_path.set_work_sample_total(list_inters)

        self.measure_time = self.conf_traffic.TIME_MIN_ACTION
        self.__interval = self.conf_traffic.TIME_MIN_ACTION

    def load_data(self, each_file):
        f_logging_data = open(each_file, "rb")
        logging_data = pickle.load(f_logging_data)
        f_logging_data.close()
        return logging_data

    def construct_state(self, logging_data, features, time):
        state = logging_data[time]
        assert time == state["time"]
        phase_expansion = self.conf_traffic.TRAFFIC_INFO['phase_lane_mapping']
        state_after_selection = {}
        for key, value in state["state"].items():
            if key in features:
                if key == "cur_phase_index":
                    state_after_selection[key] = phase_expansion[value - 1]
                else:
                    state_after_selection[key] = value
        return state_after_selection

    def __get_reward_from_features(self, rs):
        reward = dict()
        reward["sum_lane_queue_length"] = np.sum(rs["stop_vehicle_thres1"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_waiting_time"])
        reward["sum_lane_vehicle_left_cnt"] = np.sum(
            rs["lane_vehicle_left_cnt"])
        reward["sum_duration_vehicle_left"] = np.sum(
            rs["lane_vehicle_left_cnt"])
        reward["sum_stop_vehicle_thres1"] = np.sum(
            rs["stop_vehicle_thres1"])
        return reward

    def __cal_reward(self, rs, rewards_components):
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

    def construct_reward(self, logging_data, rewards_components, time):
        rs = logging_data[time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.__get_reward_from_features(rs['state'])
        r_instant = self.__cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):
            # print("t is ", t)
            rs = logging_data[t]
            assert t == rs["time"]
            rs = self.__get_reward_from_features(rs['state'])
            r = self.__cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    def judge_action(self, logging_data, time):
        action = logging_data[time]['action']
        if isinstance(action, np.ndarray) and len(action) > 1:
            return action
        elif action != -1:
            return action
        else:
            raise ValueError("sample action is a invalid value.")

    def make_reward(self):
        """round-> generator -> intersections
        Returns:
        """
        gen_cnt = self.__conf_exp.NUM_GENERATORS
        for idx, each_file in enumerate(self.__conf_path.WORK_SAMPLE_EACH):
            logging_data = self.load_data(each_file)
            total_time = int(logging_data[-1]['time'] + 1)
            list_samples = []
            for time in range(0, total_time - self.measure_time + 1,
                              self.__interval):
                state = self.construct_state(
                    logging_data, self.conf_traffic.FEATURE, time)
                reward_instant, reward_average = self.construct_reward(
                    logging_data, self.conf_traffic.REWARD_INFOS, time)
                action = self.judge_action(logging_data, time)

                if time + self.__interval == total_time:
                    next_state = self.construct_state(
                        logging_data, self.conf_traffic.FEATURE,
                        time + self.__interval - 1)
                else:
                    next_state = self.construct_state(
                        logging_data, self.conf_traffic.FEATURE,
                        time + self.__interval)
                sample = [state, action, next_state, reward_average,
                          reward_instant, time]
                list_samples.append(sample)

            self.dump_sample(
                list_samples,
                self.__conf_path.WORK_SAMPLE_TOTAL[int(idx / gen_cnt)])

    def dump_sample(self, samples, file_name):
        with open(file_name, "ab+") as f:
            pickle.dump(samples, f, -1)
