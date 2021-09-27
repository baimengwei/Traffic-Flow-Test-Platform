from misc.utils import *
import traci
import sys

os.environ['SUMO_HOME'] = '/usr/share/sumo'
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


class Intersection:
    def __init__(self, inter, dic_traffic_env_conf, eng):
        self.inter = inter
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.eng = eng

        self.lane_phase_info = self._get_inter_info()
        self.yellow_phase_index = self.lane_phase_info["yellow_phase"]
        self.list_phase_index = self.lane_phase_info["list_phase"]
        self.list_lane_enters = self.lane_phase_info["list_lane_enters"]
        self.list_lane_exits = self.lane_phase_info["list_lane_exits"]

        self.inter_id = self.lane_phase_info["inter_id"]
        self.dic_phase_strs = self._create_phase_str()

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.dic_lane_vehicle_id_pre = {}

        self._set_tl_phase(self.current_phase_index)
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_yellow_flag = False

        self.dic_vehicle_arrive_leave_time = dict()

    def get_state(self, list_state_feature):
        dic_state = {feature: self.dic_feature[feature]
                     for feature in list_state_feature}
        return dic_state

    def set_signal(self, action, yellow_time):
        if self.all_yellow_flag:
            if self.current_phase_duration >= yellow_time:  # yellow time
                self.current_phase_index = self.next_phase_to_set_index
                self._set_tl_phase(self.current_phase_index)
                self.all_yellow_flag = False
            else:
                pass
        else:
            self.next_phase_to_set_index = action + 1
            if self.current_phase_index == self.next_phase_to_set_index:
                pass
            else:
                self._set_tl_phase(self.yellow_phase_index)
                self.current_phase_index = self.yellow_phase_index
                self.all_yellow_flag = True

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

    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_id_pre = self.dic_lane_vehicle_id

    def update_current_measurements(self):
        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1
        self.lane_vehicle_cnt = [
            self.eng.lane.getLastStepVehicleNumber(lane_id)
            for lane_id in self.list_lane_enters]
        self.lane_vehicle_left_cnt = [
            self.eng.lane.getLastStepHaltingNumber(lane_id)
            for lane_id in self.list_lane_exits]
        self.lane_vehicle_waiting = [
            self.eng.lane.getLastStepHaltingNumber(lane_id)
            for lane_id in self.list_lane_enters]

        vehicles_ = []
        for lane_id in self.list_lane_enters:
            lane_vehicles = self.eng.lane.getLastStepVehicleIDs(lane_id)
            for vehicle in lane_vehicles:
                vehicles_.append(vehicle)
        self.dic_lane_vehicle_id = vehicles_

        list_vehicle_arrive = \
            set(self.dic_lane_vehicle_id) - set(self.dic_lane_vehicle_id_pre)
        list_vehicle_left = \
            set(self.dic_lane_vehicle_id_pre) - set(self.dic_lane_vehicle_id)
        self._update_vehicle_arrive_left(list_vehicle_arrive, list_vehicle_left)

        self.lane_waiting_time = [self.eng.lane.getWaitingTime(lane_id)
                                  for lane_id in self.list_lane_enters]

        self._update_feature()

    def _update_vehicle_arrive_left(self, list_arrive, list_left):
        # arrive
        ts = self.eng.simulation.getTime()
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
        dic_feature["lane_vehicle_cnt"] = self.lane_vehicle_cnt
        dic_feature["stop_vehicle_thres1"] = self.lane_vehicle_waiting
        dic_feature["lane_vehicle_left_cnt"] = self.lane_vehicle_left_cnt
        dic_feature["lane_waiting_time"] = self.lane_waiting_time
        self.dic_feature = dic_feature

    def _set_tl_phase(self, phase_index):
        if phase_index != self.yellow_phase_index:
            phase_str = self.dic_phase_strs[self.current_phase_index]
        else:
            phase_str = ''
            for i in self.dic_phase_strs[self.current_phase_index]:
                if i != 'G':
                    phase_str += 'r'
                else:
                    phase_str += 'y'
        self.eng.trafficlight.setRedYellowGreenState(
            self.inter_id, phase_str)

    def _get_inter_info(self):
        dic_lane_phase_info = {}
        list_links = self.inter.getLinks()

        list_phase = sorted(list_links.keys())
        dic_lane_phase_info["list_phase"] = list_phase

        dic_lane_phase_info["yellow_phase"] = -1
        dic_lane_phase_info["inter_id"] = self.inter.getID()

        list_roads = sorted(self.inter.getEdges(), key=lambda x: x.getID())
        dic_lane_phase_info["list_roads_enters"] = list_roads

        list_lane_enters, list_lane_exits = [], []
        for phase in list_phase:
            enter_exit_phase = list_links[phase]
            for each in enter_exit_phase:
                start, end = each[0].getID(), each[1].getID()
                if start.split('_')[-1] != end.split('_')[-1]:
                    continue
                if start not in list_lane_enters:
                    list_lane_enters.append(start)
                if end not in list_lane_exits:
                    list_lane_exits.append(end)
        dic_lane_phase_info["list_lane_enters"] = list_lane_enters
        dic_lane_phase_info["list_lane_exits"] = list_lane_exits

        return dic_lane_phase_info

    def _create_phase_str(self):
        phase_str = {}
        phase_length = len(self.list_phase_index + [self.yellow_phase_index])
        for idx in self.list_phase_index:
            each_phase = ''
            for i in range(phase_length):
                if idx == i:
                    each_phase += 'G'
                else:
                    each_phase += 'r'
            phase_str[idx] = each_phase
        return phase_str


class SumoEnv:
    def __init__(self, dic_path, dic_traffic_env_conf):
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.lane_phase_infos = self.get_lane_phase_infos()
        self.list_intersection = self.lane_phase_infos["traffic_lights"]

        self.path_to_log = self.dic_path["PATH_TO_WORK"]
        self.path_to_data = self.dic_path["PATH_TO_DATA"]
        self.yellow_time = self.dic_traffic_env_conf["YELLOW_TIME"]
        self.stop_cnt = 0

    def reset(self):
        file_sumocfg = self.dic_path["PATH_TO_ROADNET_FILE"
                       ].split(".net.xml")[0] + ".sumocfg"
        interval = str(self.dic_traffic_env_conf["INTERVAL"])
        if self.dic_traffic_env_conf['IF_GUI']:
            self.sumo_cmd = ["sumo-gui", '-c', file_sumocfg,
                             "--no-warnings", "--no-step-log",
                             "--step-length", interval]
        else:
            self.sumo_cmd = ["sumo", '-c', file_sumocfg,
                             "--no-warnings", "--no-step-log",
                             "--step-length", interval]
        print("start sumo")
        self.eng = traci
        self.eng.start(self.sumo_cmd)
        self.reset_prepare()
        state = self._get_state()
        return state

    def _get_state(self):
        list_state = [
            inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
            for inter in self.list_inter_handler]
        return list_state

    def _get_reward(self):
        list_reward = [
            inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"])
            for inter in self.list_inter_handler]
        return list_reward

    def reset_prepare(self):
        self.list_inter_handler = []
        self.list_inter_log = dict()

        for inter in self.list_intersection:
            intersection = Intersection(
                inter, self.dic_traffic_env_conf, self.eng)
            self.list_inter_handler.append(intersection)
            self.list_inter_log[inter.getID()] = []

        for inter in self.list_inter_handler:
            inter.update_current_measurements()

    def get_lane_phase_infos(self):
        lane_phase_infos = {}
        roadnet = sumolib.net.readNet(self.dic_path["PATH_TO_ROADNET_FILE"])
        traffic_lights = roadnet.getTrafficLights()
        lane_phase_infos["traffic_lights"] = traffic_lights
        return lane_phase_infos

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
            instant_time = self.eng.simulation.getTime()
            before_action_feature = self._get_feature()

            self._inner_step(action_in_sec)
            reward = self._get_reward()
            average_reward = (average_reward * i + reward[0]) / (i + 1)

            self._record_log(cur_time=instant_time,
                             before_action_feature=before_action_feature,
                             action=action_in_sec_display)
            next_state = self._get_state()
            # TODO add function here.
            if self.dic_traffic_env_conf["DONE_ENABLE"]:
                done = False
            else:
                done = False
            print('.', end='')
            if done:
                print("||done||")
        return next_state, reward, done, [average_reward]

    def bulk_log(self):
        valid_flag = {}
        for inter in self.list_inter_handler:
            inter_name = inter.inter_id
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
        self.eng.close()

    def save_replay(self):
        for inter in self.list_inter_handler:
            inter_name = inter.inter_id
            path_to_log_file = os.path.join(
                self.path_to_log, "%s.pkl" % inter_name)
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_name], f)
            f.close()
        vol = get_total_traffic_volume(
            self.dic_traffic_env_conf["TRAFFIC_FILE"])

    def _inner_step(self, action):
        for inter in self.list_inter_handler:
            inter.update_previous_measurements()
        for inter_ind, inter in enumerate(self.list_inter_handler):
            inter.set_signal(action=action[inter_ind],
                             yellow_time=self.yellow_time)

        self.eng.simulationStep()
        for inter in self.list_inter_handler:
            inter.update_current_measurements()

    def _get_feature(self):
        list_feature = [inter.dic_feature for inter in self.list_inter_handler]
        return list_feature

    def _record_log(self, cur_time, before_action_feature, action):
        for idx, inter in enumerate(self.list_intersection):
            inter_id = inter.getID()
            self.list_inter_log[inter_id].append(
                {"time": cur_time,
                 "state": before_action_feature[idx],
                 "action": action[idx]})


if __name__ == '__main__':
    os.chdir('../')
    # Out of date. refresh please.
    print('sumo env test start...')
    print('test finished..')
