from configs.config_phaser import *


class ConstructSample:
    def __init__(self, path_to_samples, round_number):

        self.path_to_samples = path_to_samples
        self.round_number = round_number

        x = os.path.join(self.path_to_samples, '../', '../', 'traffic_env.conf')
        with open(x) as f:
            self.dic_traffic_env_conf = json.load(f)

        self.parent_dir = os.path.join(self.path_to_samples, '../')

    def load_data(self, folder):
        self.measure_time = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        self.logging_data = []
        for inter_name in self.dic_traffic_env_conf["LANE_PHASE_INFOS"].keys():
            file_name = os.path.join(self.path_to_samples, folder,
                                     "%s.pkl" % inter_name)
            f_logging_data = open(file_name, "rb")
            self.logging_data.append(pickle.load(f_logging_data))
            f_logging_data.close()

    def construct_state(self, logging_data, features, time):
        state = logging_data[time]
        assert time == state["time"]
        phase_expansion = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]['phase_lane_mapping']
        state_after_selection = {}
        for key, value in state["state"].items():
            if key in features:
                if key == "cur_phase":
                    state_after_selection[key] = phase_expansion[str(value - 1)]
                else:
                    state_after_selection[key] = value
        return state_after_selection

    def get_reward_from_features(self, rs):
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

    def construct_reward(self, logging_data, rewards_components, time):
        rs = logging_data[time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])
        r_instant = self.cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):
            # print("t is ", t)
            rs = logging_data[t]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
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
        for folder in os.listdir(self.path_to_samples):
            if "generator" not in folder:
                continue

            self.load_data(folder)

            list_inters = self.dic_traffic_env_conf['LANE_PHASE_INFOS'].keys()
            for logging_data, inter_name in zip(self.logging_data, list_inters):
                self.dic_traffic_env_conf = update_traffic_env_info(
                    self.dic_traffic_env_conf, inter_name)
                list_samples = []
                total_time = int(logging_data[-1]['time'] + 1)
                # construct samples
                for time in range(0, total_time - self.measure_time + 1,
                                  self.interval):
                    state = self.construct_state(
                        logging_data,
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time)
                    reward_instant, reward_average = self.construct_reward(
                        logging_data,
                        self.dic_traffic_env_conf["DIC_REWARD_INFO"], time)
                    action = self.judge_action(logging_data, time)

                    if time + self.interval == total_time:
                        next_state = self.construct_state(
                            logging_data,
                            self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
                            , time + self.interval - 1)
                    else:
                        next_state = self.construct_state(
                            logging_data,
                            self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                            time + self.interval)
                    sample = [state, action, next_state, reward_average,
                              reward_instant, time]
                    list_samples.append(sample)
                self.dump_sample(list_samples, inter_name)

    def dump_sample(self, samples, inter_name):
        with open(os.path.join(
                self.parent_dir, "total_samples_%s.pkl" % inter_name),
                "ab+") as f:
            pickle.dump(samples, f, -1)
