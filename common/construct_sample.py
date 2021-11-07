from configs.config_phaser import *


class ConstructSample:
    def __init__(self, conf_path: ConfPath, round_number):
        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__conf_exp, _, self.__conf_traffic = self.__conf_path.load_conf_file()

        list_inters = sorted(list(self.__conf_traffic.TRAFFIC_INFOS.keys()))
        self.__conf_path.set_work_sample_each(
            self.__round_number, self.__conf_exp.NUM_GENERATORS, list_inters)

        self.__measure_time = self.__conf_traffic.MIN_ACTION_TIME
        self.__interval = self.__conf_traffic.MIN_ACTION_TIME

    def __load_data(self, each_file):
        logging_data = []
        f_logging_data = open(each_file, "rb")
        logging_data.append(pickle.load(f_logging_data))
        f_logging_data.close()
        return logging_data

    def __construct_state(self, logging_data, features, time):
        state = logging_data[time]
        assert time == state["time"]
        phase_expansion = self.__conf_traffic.TRAFFIC_INFO['phase_lane_mapping']
        state_after_selection = {}
        for key, value in state["state"].items():
            if key in features:
                if key == "cur_phase":
                    state_after_selection[key] = phase_expansion[str(value - 1)]
                else:
                    state_after_selection[key] = value
        return state_after_selection

    def __get_reward_from_features(self, rs):
        reward = dict()
        reward["sum_lane_queue_length"] = np.sum(rs["stop_vehicle_thres1"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_waiting_time"])
        reward["sum_lane_vehicle_left_cnt"] = np.sum(
            rs["lane_vehicle_left_cnt"])
        # TODO remove please.
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

    def __construct_reward(self, logging_data, rewards_components, time):
        rs = logging_data[time + self.__measure_time - 1]
        assert time + self.__measure_time - 1 == rs["time"]
        rs = self.__get_reward_from_features(rs['state'])
        r_instant = self.__cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.__measure_time):
            # print("t is ", t)
            rs = logging_data[t]
            assert t == rs["time"]
            rs = self.__get_reward_from_features(rs['state'])
            r = self.__cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    def __judge_action(self, logging_data, time):
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
            logging_data = self.__load_data(each_file)
            total_time = int(logging_data[-1]['time'] + 1)
            list_samples = []
            for time in range(0, total_time - self.__measure_time + 1,
                              self.__interval):
                state = self.__construct_state(
                    logging_data, self.__conf_traffic.FEATURE, time)
                reward_instant, reward_average = self.__construct_reward(
                    logging_data, self.__conf_traffic.DIC_REWARD_INFO, time)
                action = self.__judge_action(logging_data, time)

                if time + self.__interval == total_time:
                    next_state = self.__construct_state(
                        logging_data, self.__conf_traffic.FEATURE,
                        time + self.__interval - 1)
                else:
                    next_state = self.__construct_state(
                        logging_data, self.__conf_traffic.FEATURE,
                        time + self.__interval)
                sample = [state, action, next_state, reward_average,
                          reward_instant, time]
                list_samples.append(sample)

            self.dump_sample(
                list_samples,
                self.__conf_path.WORK_SAMPLE_TOTAL[(idx + 1) % gen_cnt])

    def dump_sample(self, samples, file_name):
        with open(file_name, "ab+") as f:
            pickle.dump(samples, f, -1)
