import numpy as np

from envs.env_base import EnvBase
from misc.utils import *
import traci
import sys

os.environ['SUMO_HOME'] = '/usr/share/sumo'
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


class Intersection:
    def __init__(self, inter_id, conf_traffic, eng):
        self.inter_id = inter_id
        # TODO try copy obj
        self.conf_traffic = conf_traffic
        self.eng = eng

        self.conf_traffic.set_intersection(self.inter_id)

        self.traffic_info = self.conf_traffic.TRAFFIC_INFO
        self.yellow_phase_index = self.traffic_info["yellow_phase"]
        self.list_phase_index = list(self.traffic_info["phase_links"].keys())
        self.list_lane_enters = self.traffic_info["list_lane_enters"]
        self.list_lane_exits = self.traffic_info["list_lane_exits"]
        self.dic_phase_strs = self.traffic_info["phase_str"]

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
        dic_feature["cur_phase_index"] = self.current_phase_index
        dic_feature["time_this_phase"] = self.current_phase_duration
        dic_feature["lane_vehicle_cnt"] = self.lane_vehicle_cnt
        dic_feature["stop_vehicle_thres1"] = self.lane_vehicle_waiting
        dic_feature["lane_vehicle_left_cnt"] = self.lane_vehicle_left_cnt
        dic_feature["lane_waiting_time"] = self.lane_waiting_time
        self.dic_feature = dic_feature

    def _set_tl_phase(self, phase_index):
        if phase_index != self.yellow_phase_index:
            phase_str = self.dic_phase_strs[self.current_phase_index - 1]
        else:
            phase_str = ''
            for i in self.dic_phase_strs[self.current_phase_index - 1]:
                if i != 'G':
                    phase_str += 'r'
                else:
                    phase_str += 'y'
        self.eng.trafficlight.setRedYellowGreenState(
            self.inter_id, phase_str)


class SumoEnv(EnvBase):
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
        file_sumocfg = self.conf_path.ROADNET_FILE.split(".net.xml")[0] + ".sumocfg"
        sumo_cmd = ["sumo", '-c', file_sumocfg,
                    "--no-warnings", "--no-step-log",
                    "--step-length", '1']
        self.eng = traci

        self.eng.start(sumo_cmd)
        self.__get_agent_info()
        # update traffic infos
        self.conf_traffic.set_traffic_infos(self.traffic_infos)
        self.yellow_time = self.conf_traffic.TIME_YELLOW
        return self.traffic_infos

    def __get_agent_info(self):
        traffic_infos = OrderedDict()
        traffic_lights = self.eng.trafficlight
        tls_ids = traffic_lights.getIDList()

        for tls_id in tls_ids:
            # bound
            traffic_info = OrderedDict()
            traffic_infos[tls_id] = traffic_info
            # phase_lane_mapping
            signals = traffic_lights.getCompleteRedYellowGreenDefinition(tls_id)
            signals = signals[0]
            phases = traffic_lights.Logic.getPhases(signals)
            phase_map = {}
            for idx, phase in enumerate(phases):
                phase_lane = [0 for _ in range(len(phase.state))]
                for i, l in enumerate(phase.state):
                    if l == 'g' or l == 'G':
                        phase_lane[i] = 1
                phase_map[idx] = phase_lane
            traffic_info['phase_lane_mapping'] = phase_map
            # phase_links
            phase_links = traffic_lights.getControlledLinks(tls_id)
            phase_links_ = OrderedDict()
            for idx, pl in enumerate(phase_links):
                if len(pl) > 0:
                    phase_links_[idx] = pl
            phase_links = phase_links_
            traffic_info['phase_links'] = phase_links
            # relation
            relation = None
            # TODO implement relation
            traffic_info['relation'] = relation
            # phase_str, yellow_phase
            signals = traffic_lights.getCompleteRedYellowGreenDefinition(tls_id)[0]
            phase_str = {idx: p.state for idx, p in enumerate(signals.phases)}
            traffic_info["phase_str"] = phase_str
            traffic_info["yellow_phase"] = -1
            # enters, exits
            list_lane_enters = [inter[0][0] for _, inter in phase_links.items()]
            list_lane_exits = [inter[0][1] for _, inter in phase_links.items()]
            traffic_info["list_lane_enters"] = list_lane_enters
            traffic_info["list_lane_exits"] = list_lane_exits

        self.traffic_infos = traffic_infos

    def reset(self):
        self.list_intersection = list(self.eng.trafficlight.getIDList())

        self.__reset_prepare()
        state = self.__get_state()
        return state

    def __reset_prepare(self):
        self.list_inter_handler = []
        self.list_inter_log = dict()

        for inter_id in self.list_intersection:
            intersection = Intersection(
                inter_id, self.conf_traffic, self.eng)
            self.list_inter_handler.append(intersection)

            self.list_inter_log[inter_id] = []

        for inter in self.list_inter_handler:
            inter.update_current_measurements()

    def __get_state(self):
        list_state = [
            inter.get_state(self.conf_traffic.FEATURE)
            for inter in self.list_inter_handler]
        return list_state

    def __get_reward(self):
        list_reward = [
            inter.get_reward(self.conf_traffic.REWARD_INFOS)
            for inter in self.list_inter_handler]
        return list_reward

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
            instant_time = self.eng.simulation.getTime()
            before_action_feature = self.__get_feature()

            self.__inner_step(action_in_sec)
            reward = self.__get_reward()
            average_reward = (average_reward * i + reward[0]) / (i + 1)

            self.__record_log(cur_time=instant_time,
                              before_action_feature=before_action_feature,
                              action=action_in_sec_display)
            next_state = self.__get_state()

            # TODO add function to make done enable running.
            if self.conf_traffic.DONE_ENABLE:
                done = False
            else:
                done = False
            if i % 100 == 0: print('.', end='')
            if done:
                print("||done||")
        return next_state, reward, done, [average_reward]

    def bulk_log(self, reward):
        valid_flag = {}
        for inter in self.list_inter_handler:
            inter_name = inter.inter_id
            path_to_log_file = os.path.join(
                self.path_to_work, "vehicle_inter_%s.csv" % inter_name)

            dic_vehicle = inter.dic_vehicle_arrive_leave_time
            df = convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

            feature = inter.dic_feature

            if max(feature['lane_vehicle_cnt']) > \
                    self.conf_traffic.VALID_THRESHOLD:
                valid_flag[inter_name] = 0
            else:
                valid_flag[inter_name] = 1
            valid_flag['%s_reward' % inter_name] = reward[inter_name]
        json.dump(valid_flag,
                  open(os.path.join(self.path_to_work, "valid_flag.json"), "w"))
        self.__save_replay()
        self.eng.close()

    def __save_replay(self):
        for inter in self.list_inter_handler:
            inter_name = inter.inter_id
            path_to_log_file = os.path.join(
                self.path_to_work, "%s.pkl" % inter_name)
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_name], f)
            f.close()

    def __inner_step(self, action):
        for inter in self.list_inter_handler:
            inter.update_previous_measurements()
        for inter_ind, inter in enumerate(self.list_inter_handler):
            inter.set_signal(action=action[inter_ind],
                             yellow_time=self.yellow_time)

        self.eng.simulationStep()
        for inter in self.list_inter_handler:
            inter.update_current_measurements()

    def __get_feature(self):
        list_feature = [inter.dic_feature for inter in self.list_inter_handler]
        return list_feature

    def __record_log(self, cur_time, before_action_feature, action):
        for idx, inter_id in enumerate(self.list_intersection):
            self.list_inter_log[inter_id].append(
                {"time": cur_time,
                 "state": before_action_feature[idx],
                 "action": action[idx]})


if __name__ == '__main__':
    os.chdir('../')
    # Out of date. refresh please.
    print('sumo env test start...')
    print('test finished..')
