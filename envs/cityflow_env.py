import json
import uuid

from misc.utils import *
import time
import cityflow
from envs.env_base import EnvBase


class Intersection:
    def __init__(self, inter_id, conf_traffic, eng):
        self.inter_id = inter_id
        # TODO try copy obj
        self.conf_traffic = conf_traffic
        self.eng = eng
        self.conf_traffic.set_intersection(self.inter_id)
        #
        self.traffic_info = self.conf_traffic.TRAFFIC_INFO
        self.yellow_phase_index = self.traffic_info["yellow_phase"]
        self.list_lane_enters = self.traffic_info["list_lane_enters"]
        self.list_lane_exits = self.traffic_info["list_lane_exits"]
        self.list_lanes = self.list_lane_enters + self.list_lane_exits

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

        self.set_tl_phase(self.inter_id, self.current_phase_index)
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_yellow_flag = False

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

    def set_signal(self, action, yellow_time):
        if self.all_yellow_flag:
            if self.current_phase_duration >= yellow_time:  # yellow time
                self.current_phase_index = self.next_phase_to_set_index
                self.set_tl_phase(self.inter_id, self.current_phase_index)
                self.all_yellow_flag = False
            else:
                pass
        else:
            self.next_phase_to_set_index = action + 1
            if self.current_phase_index == self.next_phase_to_set_index:
                pass
            else:
                self.set_tl_phase(self.inter_id, self.yellow_phase_index)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True

    def set_tl_phase(self, inter_id, phase_index):
        """API in different environment
        """
        self.eng.set_tl_phase(inter_id, phase_index)

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
            for lane in self.list_lane_enters:
                list_entering_lane_vehicle_left.append([])
        else:
            for lane in self.list_lane_enters:
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
        dic_feature["cur_phase_index"] = self.current_phase_index
        dic_feature["time_this_phase"] = self.current_phase_duration

        dic_feature["lane_vehicle_cnt"] = \
            [len(self.dic_lane_vehicle_current_step[lane])
             for lane in self.list_lane_enters]
        dic_feature["stop_vehicle_thres1"] = \
            [self.dic_lane_vehicle_waiting_current_step[lane]
             for lane in self.list_lane_enters]
        dic_feature["lane_queue_length"] = \
            [self.dic_lane_vehicle_waiting_current_step[lane]
             for lane in self.list_lane_enters]
        dic_feature["lane_vehicle_left_cnt"] = \
            [len(self.dic_lane_vehicle_current_step[lane])
             for lane in self.list_lane_exits]

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


class CityflowEnv(EnvBase):
    def __init__(self, conf_path, *, is_test=False):
        self.conf_path = conf_path
        _, _, self.conf_traffic = self.conf_path.load_conf_file()
        if is_test:
            self.path_to_work = self.conf_path.WORK_TEST
        else:
            self.path_to_work = self.conf_path.WORK_SAMPLE
        self.path_to_data = self.conf_path.DATA
        self.stop_cnt = 0

    def get_agents_info(self):
        self.traffic_infos = \
            self.__get_agents_info(self.conf_path.ROADNET_FILE)

        self.conf_traffic.set_traffic_infos(self.traffic_infos)
        self.yellow_time = self.conf_traffic.TIME_YELLOW
        return self.traffic_infos

    def __get_agents_info(self, roadnet_file_dir):
        """DISGUSTING code.
        Args:
            roadnet_file_dir: a full dir of the roadnet file.
        Returns:
            file infos
        """
        with open(roadnet_file_dir) as f:
            roadnet = json.load(f)
        traffic_infos = OrderedDict()
        list_inter = [inter for inter in roadnet["intersections"]
                      if not inter["virtual"]]

        for inter in list_inter:
            # bound
            traffic_info = OrderedDict()
            traffic_infos[inter['id']] = traffic_info
            # phase_lane_mapping
            phase_lane = inter['trafficLight']['lightphases']
            list_indices = inter['trafficLight']['roadLinkIndices']
            phase_lane_ = OrderedDict()
            for idx, lanes in enumerate(phase_lane):
                list_links = lanes['availableRoadLinks']
                if len(list_links) > 0:
                    list_lane = [0 for _ in range(len(list_indices))]
                    for ll in list_links:
                        list_lane[ll] = 1
                    phase_lane_[idx - 1] = list_lane
            phase_lane = phase_lane_
            traffic_info['phase_lane_mapping'] = phase_lane
            # phase_links
            phase_links_ = OrderedDict()
            phase_links = inter['roadLinks']
            for idx, pl in enumerate(phase_links):
                lane_s = pl['startRoad']
                lane_e = pl['endRoad']
                lane_links = pl['laneLinks']
                si, ei = None, None
                for ll in lane_links:
                    si = ll['startLaneIndex']
                    ei = ll['endLaneIndex']
                    if si == ei:
                        lane_s = ''.join([lane_s, '_', str(si)])
                        lane_e = ''.join([lane_e, '_', str(ei)])
                        break
                else:
                    lane_s = ''.join([lane_s, '_', str(si)])
                    lane_e = ''.join([lane_e, '_', str(ei)])
                phase_links_[idx] = [lane_s, lane_e]
            phase_links = phase_links_
            traffic_info['phase_links'] = phase_links
            # relation
            relation = get_relation(phase_lane)
            traffic_info['relation'] = relation
            # phase_str, yellow_phase
            # note that the phase_str is not used for cityflow
            traffic_info['phase_str'] = None
            traffic_info["yellow_phase"] = 0
            # enters, exits
            lane_enters = [pl[0] for _, pl in phase_links.items()]
            lane_exits = [pl[1] for _, pl in phase_links.items()]
            traffic_info['list_lane_enters'] = lane_enters
            traffic_info['list_lane_exits'] = lane_exits
        return traffic_infos

    def reset(self):
        if not os.path.isfile(self.conf_path.ROADNET_FILE):
            raise FileExistsError("file not exist! check it %s" %
                                  self.conf_path.ROADNET_FILE)
        if not os.path.isfile(self.conf_path.FLOW_FILE):
            raise FileExistsError("file not exist! check it %s" %
                                  self.conf_path.FLOW_FILE)
        config_file = self.__save_config_file()

        self.eng = cityflow.Engine(config_file, 1)
        print(config_file)
        os.remove(config_file)
        self.__reset_prepare()
        state = self.__get_state()
        return state

    def __reset_prepare(self):
        self.list_intersection = []
        self.list_inter_log = dict()

        for inter_id in sorted(self.traffic_infos.keys()):
            intersection = Intersection(inter_id, self.conf_traffic, self.eng)
            self.list_intersection.append(intersection)
            self.list_inter_log[inter_id] = []

        for inter in self.list_intersection:
            inter.update_current_measurements()

    def step(self, action):
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.conf_traffic.TIME_MIN_ACTION - 1):
            list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(
                np.full_like(action, fill_value=-1).tolist())
        average_reward = 0
        next_state, reward, done = None, None, None
        for i in range(self.conf_traffic.TIME_MIN_ACTION):
            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]
            instant_time = self.__get_current_time()
            before_action_feature = self.__get_feature()

            self.__inner_step(action_in_sec)
            reward = self.__get_reward()
            average_reward = (average_reward * i + reward[0]) / (i + 1)

            self.__log(cur_time=instant_time,
                       before_action_feature=before_action_feature,
                       action=action_in_sec_display)
            next_state = self.__get_state()
            if self.conf_traffic.DONE_ENABLE:
                done = self._check_episode_done(next_state)
            else:
                done = False
            if done:
                print("||done||")
        return next_state, reward, done, [average_reward]

    def __inner_step(self, action):
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(action=action[inter_ind],
                             yellow_time=self.yellow_time)
        self.eng.next_step()
        for inter in self.list_intersection:
            inter.update_current_measurements()

    def __check_episode_done(self, state):
        if 39 in state[0]["lane_vehicle_cnt"]:
            self.stop_cnt += 1
        if self.stop_cnt == 100:
            self.stop_cnt = 0
            return True
        else:
            return False

    def __get_feature(self):
        list_feature = [inter.dic_feature for inter in self.list_intersection]
        return list_feature

    def __get_state(self):
        list_state = [inter.get_state(self.conf_traffic.FEATURE)
                      for inter in self.list_intersection]
        return list_state

    def __get_reward(self):
        list_reward = [inter.get_reward(self.conf_traffic.REWARD_INFOS)
                       for inter in self.list_intersection]
        return list_reward

    def __log(self, cur_time, before_action_feature, action):
        for idx, inter_id in enumerate(sorted(self.traffic_infos.keys())):
            self.list_inter_log[inter_id].append(
                {"time": cur_time,
                 "state": before_action_feature[idx],
                 "action": action[idx]})

    def bulk_log(self, reward):
        valid_flag = {}
        for inter in self.list_intersection:
            inter_id = inter.inter_id
            path_to_log_file = os.path.join(
                self.path_to_work, "vehicle_inter_%s.csv" % inter_id)
            dic_vehicle = inter.dic_vehicle_arrive_leave_time
            df = convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

            feature = inter.dic_feature

            if max(feature['lane_vehicle_cnt']) > \
                    self.conf_traffic.VALID_THRESHOLD:
                valid_flag[inter_id] = 0
            else:
                valid_flag[inter_id] = 1
            valid_flag['%s_reward' % inter_id] = reward[inter_id]
        json.dump(valid_flag,
                  open(os.path.join(self.path_to_work, "valid_flag.json"), "w"))
        self.__save_replay()

    def __log_phase(self):
        for inter in self.list_intersection:
            print(
                "%f, %f" %
                (self.get_current_time(), inter.current_phase_index),
                file=open(os.path.join(self.path_to_work, "log_phase.txt"), "a"))

    def __get_current_time(self):
        return self.eng.get_current_time()

    def __save_config_file(self):
        config_dict = {
            "interval": 1,
            "seed": 0,
            "dir": "",
            "roadnetFile": self.conf_path.ROADNET_FILE,
            "flowFile": self.conf_path.FLOW_FILE,
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile": None,
            "replayLogFile": None,
        }
        config_name = str(time.time_ns()) + str(uuid.uuid1()) + ".tmp"

        with open(config_name, "w") as f:
            # f.flush()
            json.dump(config_dict, f)
            # os.fsync(f)
        return config_name

    def __save_replay(self):
        for inter_id in sorted(self.traffic_infos.keys()):
            path_to_log_file = os.path.join(
                self.path_to_work, "%s.pkl" % inter_id)
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_id], f)
            f.close()


if __name__ == '__main__':
    os.chdir('../')
    # Out of date. refresh please.
    print('cityflow env test start...')

    from configs import config_phaser

    args = config_phaser.parse()
    conf_exp, conf_agent, conf_traffic, conf_path = \
        config_phaser.config_all(args)

    conf_path.set_traffic_file('hangzhou_baochu_tiyuchang_1h_10_11_2021')
    conf_path.create_path_dir()
    conf_path.dump_conf_file(conf_exp, conf_agent, conf_traffic)
    env = CityflowEnv(conf_path)
    traffic_info = env.get_agents_info()
    print('----------traffic_info-------------')
    print( json.dumps(traffic_info))
    print('----------------------------------')
    state = env.reset()
    done = False
    cnt = 0
    while not done and cnt < 360:
        cnt += 1
        action = [1]
        next_state, reward, done, _ = env.step(action)
        print(state, action, reward, next_state, done, _)
        state = next_state

    print('test finished..')
