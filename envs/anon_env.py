from misc.utils import *
import sys
import platform

if platform.system() == "Linux":
    import engine


class Intersection:
    def __init__(self, inter_name, dic_traffic_env_conf, eng):
        self.inter_name = inter_name
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.eng = eng

        self.dic_traffic_env_conf["LANE_PHASE_INFO"] = \
            self.dic_traffic_env_conf["LANE_PHASE_INFOS"][self.inter_name]
        self.list_entering_lanes = \
            self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
        self.list_exiting_lanes = \
            self.dic_traffic_env_conf["LANE_PHASE_INFO"]["end_lane"]
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes
        self.yellow_phase_index = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]["yellow_phase"]

        # previous & current
        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_waiting_current_step = {}
        self.dic_lane_vehicle_waiting_previous_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        # -1: all yellow
        self.all_yellow_phase_index = -1

        self.set_tl_phase(self.inter_name, self.current_phase_index)
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_yellow_flag = False

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

    def set_signal(self, action, yellow_time):
        if self.all_yellow_flag:
            if self.current_phase_duration >= yellow_time:  # yellow time
                self.current_phase_index = self.next_phase_to_set_index
                self.set_tl_phase(self.inter_name, self.current_phase_index)
                self.all_yellow_flag = False
            else:
                pass
        else:
            self.next_phase_to_set_index = action + 1
            if self.current_phase_index == self.next_phase_to_set_index:
                pass
            else:
                self.set_tl_phase(self.inter_name, self.yellow_phase_index)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True

    def set_tl_phase(self, inter_name, phase_index):
        """API in different environment
        """
        self.eng.set_tl_phase(inter_name, phase_index)

    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = \
            self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_waiting_previous_step = \
            self.dic_lane_vehicle_waiting_current_step
        self.dic_vehicle_speed_previous_step = \
            self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = \
            self.dic_vehicle_distance_current_step

    def update_current_measurements(self):
        # DISGUSTING code for these names
        # same to __init__ function values. AND UPDATE FEATURE etc.
        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1
        self.dic_lane_vehicle_current_step = self.eng.get_lane_vehicles()
        self.dic_lane_vehicle_waiting_current_step = \
            self.eng.get_lane_waiting_vehicle_count()
        if self.dic_traffic_env_conf["ENV_DEBUG"]:
            self.dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
            self.dic_vehicle_distance_current_step = \
                self.eng.get_vehicle_distance()

        vehicle_now = get_vehicle_list(self.dic_lane_vehicle_current_step)
        vehicle_pre = get_vehicle_list(self.dic_lane_vehicle_previous_step)
        list_vehicle_new_arrive = list(set(vehicle_now) - set(vehicle_pre))
        #  this maybe the true value. the function below maybe
        #  think the vehicle leave the lane of entering is leaving the env

        # the comment think the vehicle leave the env is in the leaving env
        # if for multi intersection, the former maybe better.
        # and get lower value
        # list_vehicle_new_left = list(set(vehicle_pre) - set(vehicle_now))

        list_entering_lane_vehicle_left = \
            self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_entering_lane_vehicle_left:
            list_vehicle_new_left_entering_lane += l

        self._update_vehicle_arrive_left(list_vehicle_new_arrive,
                                         list_vehicle_new_left_entering_lane)
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append(
                    list(
                        set(self.dic_lane_vehicle_previous_step[lane]) - \
                        set(self.dic_lane_vehicle_current_step[lane])
                    )
                )
        return list_entering_lane_vehicle_left

    def _update_vehicle_arrive_left(self, list_arrive, list_left):
        # arrive
        ts = self.get_current_time()
        for vehicle in list_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                pass
        # left
        for vehicle in list_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_feature(self):
        dic_feature = dict()

        dic_feature["cur_phase"] = self.current_phase_index
        dic_feature["time_this_phase"] = self.current_phase_duration

        dic_feature["lane_vehicle_cnt"] = \
            [len(self.dic_lane_vehicle_current_step[lane]) for lane in
             self.list_entering_lanes]
        dic_feature["stop_vehicle_thres1"] = \
            [self.dic_lane_vehicle_waiting_current_step[lane]
             for lane in self.list_entering_lanes]
        dic_feature["lane_queue_length"] = \
            [self.dic_lane_vehicle_waiting_current_step[lane]
             for lane in self.list_entering_lanes]
        dic_feature["lane_vehicle_left_cnt"] = \
            [len(self.dic_lane_vehicle_current_step[lane]) for lane in
             self.list_exiting_lanes]

        dic_feature["lane_duration_vehicle_left"] = None
        dic_feature["lane_waiting_time"] = None
        dic_feature["terminal"] = None
        self.dic_feature = dic_feature

    # ------------------- not used now---------------------------------
    def _get_lane_vehicle_speed(self, list_lanes):
        return [self.dic_vehicle_speed_current_step[lane]
                for lane in list_lanes]

    # ================= get functions from outside ======================
    def get_state(self, list_state_features):
        dic_state = {feature: self.dic_feature[feature]
                     for feature in list_state_features}
        return dic_state

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_vehicle_left_cnt"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_stop_vehicle_thres1"] = \
            np.sum(self.dic_feature["stop_vehicle_thres1"])
        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward

    def get_current_time(self):
        # a API for different env.
        return self.eng.get_current_time()


class AnonEnv:
    def __init__(self, dic_path, dic_traffic_env_conf):
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.path_to_log = self.dic_path["PATH_TO_WORK"]
        self.path_to_data = self.dic_path["PATH_TO_DATA"]
        self.lane_phase_infos = self.dic_traffic_env_conf['LANE_PHASE_INFOS']
        self.yellow_time = self.dic_traffic_env_conf["YELLOW_TIME"]
        self.stop_cnt = 0

    def reset(self):
        self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
                                 self.dic_traffic_env_conf["THREADNUM"],
                                 self.dic_traffic_env_conf["SAVEREPLAY"],
                                 self.dic_traffic_env_conf["RLTRAFFICLIGHT"])
        if not os.path.isfile(self.dic_path["PATH_TO_ROADNET_FILE"]):
            raise FileExistsError("file not exist! check it %s" %
                                  self.dic_path["PATH_TO_ROADNET_FILE"])
        if not os.path.isfile(self.dic_path["PATH_TO_FLOW_FILE"]):
            raise FileExistsError("file not exist! check it %s" %
                                  self.dic_path["PATH_TO_FLOW_FILE"])

        self.eng.load_roadnet(self.dic_path["PATH_TO_ROADNET_FILE"])
        self.eng.load_flow(self.dic_path["PATH_TO_FLOW_FILE"])

        self.reset_prepare()
        state = self.get_state()
        return state

    def reset_prepare(self):
        self.list_intersection = []
        self.list_inter_log = dict()
        self.list_lanes = []
        for inter_name in sorted(self.lane_phase_infos.keys()):
            intersection = Intersection(
                inter_name, self.dic_traffic_env_conf, self.eng)
            self.list_intersection.append(intersection)
            self.list_inter_log[inter_name] = []
            self.list_lanes += intersection.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        for inter in self.list_intersection:
            inter.update_current_measurements()

    def step(self, action):
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"] - 1):
            list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(
                np.full_like(action, fill_value=-1).tolist())
        average_reward = 0
        next_state, reward, done = None, None, None
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]
            instant_time = self.get_current_time()
            before_action_feature = self.get_feature()

            self._inner_step(action_in_sec)
            reward = self.get_reward()
            average_reward = (average_reward * i + reward[0]) / (i + 1)

            self.log(cur_time=instant_time,
                     before_action_feature=before_action_feature,
                     action=action_in_sec_display)
            next_state = self.get_state()
            if self.dic_traffic_env_conf["DONE_ENABLE"]:
                done = self._check_episode_done(next_state)
            else:
                done = False
            print('.', end='')
            if done:
                print("||done||")
        return next_state, reward, done, [average_reward]

    def _inner_step(self, action):
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(action=action[inter_ind],
                             yellow_time=self.yellow_time)
        for i in range(int(1 / self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()
        for inter in self.list_intersection:
            inter.update_current_measurements()
        if self.dic_traffic_env_conf["ENV_DEBUG"]:
            self.log_phase()

    def _check_episode_done(self, state):
        if 39 in state[0]["lane_vehicle_cnt"]:
            self.stop_cnt += 1
        if self.stop_cnt == 100:
            self.stop_cnt = 0
            return True
        else:
            return False

    def get_feature(self):
        list_feature = [inter.dic_feature for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        list_state = [
            inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
            for inter in self.list_intersection]
        return list_state

    def get_reward(self):
        list_reward = [
            inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for
            inter in self.list_intersection]
        return list_reward

    def log(self, cur_time, before_action_feature, action):
        for idx, inter_name in enumerate(sorted(self.lane_phase_infos.keys())):
            self.list_inter_log[inter_name].append(
                {"time": cur_time,
                 "state": before_action_feature[idx],
                 "action": action[idx]})

    def bulk_log(self):
        valid_flag = {}
        for inter in self.list_intersection:
            inter_name = inter.inter_name
            path_to_log_file = os.path.join(
                self.path_to_log, "vehicle_inter_%s.csv" % inter_name)
            dic_vehicle = inter.dic_vehicle_arrive_leave_time
            df = convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

            feature = inter.dic_feature

            if max(feature['lane_vehicle_cnt']) > \
                    self.dic_traffic_env_conf["VALID_THRESHOLD"]:
                valid_flag[inter_name] = 0
            else:
                valid_flag[inter_name] = 1
        json.dump(valid_flag,
                  open(os.path.join(self.path_to_log, "valid_flag.json"), "w"))
        self.save_replay()

    def save_replay(self):
        for inter_name in sorted(self.lane_phase_infos.keys()):
            path_to_log_file = os.path.join(
                self.path_to_log, "%s.pkl" % inter_name)
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_name], f)
            f.close()
        vol = get_total_traffic_volume(
            self.dic_traffic_env_conf["TRAFFIC_FILE"])
        self.eng.print_log(
            os.path.join(self.path_to_log, "roadnet_1_1.json"),
            os.path.join(self.path_to_log, "replay_1_1_%s.txt" % vol))

    def log_phase(self):
        for inter in self.list_intersection:
            print(
                "%f, %f" %
                (self.get_current_time(), inter.current_phase_index),
                file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))

    def get_current_time(self):
        return self.eng.get_current_time()


if __name__ == '__main__':
    os.chdir('../')
    # Out of date. refresh please.
    print('anon env test start...')
    from configs.config_example import *
    from configs.config_phaser import *

    create_path_dir(dic_path)
    env = AnonEnv(dic_path, dic_traffic_env)
    state = env.reset()
    done = False
    cnt = 0
    while not done and cnt < 360:
        cnt += 1
        action = [0]
        next_state, reward, done, _ = env.step(action)
        print(state, action, reward, next_state, done, _)
        state = next_state

    print('test finished..')
