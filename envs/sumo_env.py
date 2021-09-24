import os
import sys
from envs.anon_env import AnonEnv
import traci
import traci.constants as tc

os.environ['SUMO_HOME'] = '/usr/share/sumo'
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


def get_traci_constant_mapping(constant_str):
    return getattr(tc, constant_str)


class Intersection:
    def __init__(self, inter_name, dic_traffic_env_conf, eng):
        self.inter_name = inter_name
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.eng = eng

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)


        # generate signals
        self.list_phases = dic_sumo_env_conf["PHASE"]
        self.dic_app_offset = {str(i): int(i) for i in self.list_approachs}

        self.dic_phase_strs = {}

        for p in self.list_phases:
            list_default_str = ["r" for i in
                                range(self.num_lane * len(self.list_approachs))]

            # set green for right turn
            for any_app in self.list_approachs:
                for ind_this_direc in \
                        np.where(np.array(self.lane_direc) == "r")[0].tolist():
                    list_default_str[self.dic_app_offset[
                                         any_app] * self.num_lane +
                                     ind_this_direc] = 'g'

            app1 = p[0]
            direc1 = p[1]
            app2 = p[3]
            direc2 = p[4]

            for ind_this_direc in \
                    np.where(np.array(self.lane_direc) == direc1.lower())[
                        0].tolist():
                list_default_str[self.dic_app_offset[
                                     app1] * self.num_lane + ind_this_direc] \
                    = 'G'
            for ind_this_direc in \
                    np.where(np.array(self.lane_direc) == direc2.lower())[
                        0].tolist():
                list_default_str[self.dic_app_offset[
                                     app2] * self.num_lane + ind_this_direc] \
                    = 'G'
            self.dic_phase_strs[p] = "".join(list_default_str)

        self.all_yellow_phase_str = "".join(
            ["y" for i in range(self.num_lane * len(self.list_approachs))])
        self.all_red_phase_str = "".join(
            ["r" for i in range(self.num_lane * len(self.list_approachs))])

        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        # initialization

        # -1: all yellow, -2: all red, -3: none
        self.current_phase_index = 0
        self.previous_phase_index = 0
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_lane_sub_current_step = None
        self.dic_lane_sub_previous_step = None
        self.dic_vehicle_sub_current_step = None
        self.dic_vehicle_sub_previous_step = None
        self.list_vehicles_current_step = []
        self.list_vehicles_previous_step = []

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second

    def set_signal(self, action, yellow_time):

        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time
                # reached
                self.current_phase_index = self.next_phase_to_set_index
                traci.trafficlights.setRedYellowGreenState(
                    self.node_light, self.dic_phase_strs[
                        self.list_phases[self.current_phase_index]])
                self.all_yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (
                                                           self.current_phase_index + 1) % len(
                        self.list_phases)
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                self.next_phase_to_set_index = action

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:  #
                # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                traci.trafficlights.setRedYellowGreenState(
                    self.node_light, self.all_yellow_phase_str)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    def update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.dic_lane_sub_previous_step = self.dic_lane_sub_current_step
        self.dic_vehicle_sub_previous_step = self.dic_vehicle_sub_current_step
        self.list_vehicles_previous_step = self.list_vehicles_current_step

    def update_current_measurements(self):
        ## need change, debug in seeing format

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        # ====== lane level observations =======

        self.dic_lane_sub_current_step = {
            lane: traci.lane.getSubscriptionResults(lane) for lane in
            self.list_lanes}

        # ====== vehicle level observations =======

        # get vehicle list
        self.list_vehicles_current_step = traci.vehicle.getIDList()
        list_vehicles_new_arrive = list(
            set(self.list_vehicles_current_step) - set(
                self.list_vehicles_previous_step))
        list_vehicles_new_left = list(
            set(self.list_vehicles_previous_step) - set(
                self.list_vehicles_current_step))
        list_vehicles_new_left_entering_lane_by_lane = \
            self._update_leave_entering_approach_vehicle()
        list_vehicles_new_left_entering_lane = []
        for l in list_vehicles_new_left_entering_lane_by_lane:
            list_vehicles_new_left_entering_lane += l

        # update subscriptions
        for vehicle in list_vehicles_new_arrive:
            traci.vehicle.subscribe(vehicle, [getattr(tc, var) for var in
                                              self.list_vehicle_variables_to_sub])

        # vehicle level observations
        self.dic_vehicle_sub_current_step = {
            vehicle: traci.vehicle.getSubscriptionResults(vehicle) for vehicle
            in self.list_vehicles_current_step}

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicles_new_arrive)
        self._update_left_time(list_vehicles_new_left_entering_lane)

        # update vehicle minimum speed in history
        self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    # ================= update current step measurements ======================

    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if self.dic_lane_sub_previous_step is None:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append(
                    list(
                        set(self.dic_lane_sub_previous_step[lane][
                                get_traci_constant_mapping(
                                    "LAST_STEP_VEHICLE_ID_LIST")]) - \
                        set(self.dic_lane_sub_current_step[lane][
                                get_traci_constant_mapping(
                                    "LAST_STEP_VEHICLE_ID_LIST")])
                    )
                )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicles_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicles_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                print("vehicle already exists!")
                sys.exit(-1)

    def _update_left_time(self, list_vehicles_left):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicles_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_vehicle_min_speed(self):
        '''
        record the minimum speed of one vehicle so far
        :return:
        '''
        dic_result = {}
        for vec_id, vec_var in self.dic_vehicle_sub_current_step.items():
            speed = vec_var[get_traci_constant_mapping("VAR_SPEED")]
            if vec_id in self.dic_vehicle_min_speed:  # this vehicle appeared
                # in previous time stamps:
                dic_result[vec_id] = min(speed,
                                         self.dic_vehicle_min_speed[vec_id])
            else:
                dic_result[vec_id] = speed
        self.dic_vehicle_min_speed = dic_result

    def _update_feature(self):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None  #
        # self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None  #
        # self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None  #
        # self._get_lane_vehicle_accumulated_waiting_time(
        # self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(
            self.list_entering_lanes)
        dic_feature[
            "stop_vehicle_thres1"] = self._get_lane_num_vehicle_been_stopped(1,
                                                                             self.list_entering_lanes)
        dic_feature["lane_queue_length"] = self._get_lane_queue_length(
            self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = self._get_lane_sum_waiting_time(
            self.list_entering_lanes)

        dic_feature["terminal"] = None

        self.dic_feature = dic_feature

    # ================= calculate features from current observations
    # ======================

    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping(
            "LAST_STEP_VEHICLE_HALTING_NUMBER")]
                for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][
                    get_traci_constant_mapping("LAST_STEP_VEHICLE_NUMBER")]
                for lane in list_lanes]

    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][
                    get_traci_constant_mapping("VAR_WAITING_TIME")]
                for lane in list_lanes]

    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''

        return None

    def _get_lane_num_vehicle_left(self, list_lanes):

        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for
                                      lane_vehicle_left in
                                      list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):

        ## not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):

        list_num_of_vec_ever_stopped = []
        for lane in list_lanes:
            cnt_vec = 0
            list_vec_id = self.dic_lane_sub_current_step[lane][
                get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                if self.dic_vehicle_min_speed[vec] < thres:
                    cnt_vec += 1
            list_num_of_vec_ever_stopped.append(cnt_vec)

        return list_num_of_vec_ever_stopped

    def _get_position_grid_along_lane(self, vec):
        pos = int(self.dic_vehicle_sub_current_step[vec][
                      get_traci_constant_mapping("VAR_LANEPOSITION")])
        return min(pos // self.length_grid, self.num_grid)

    def _get_lane_vehicle_position(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_sub_current_step[lane][
                get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_speed(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][
                get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][
                    get_traci_constant_mapping("VAR_SPEED")]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][
                get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][
                    get_traci_constant_mapping("VAR_ACCUMULATED_WAITING_TIME")]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    # ================= get functions from outside ======================

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def get_dic_vehicle_arrive_leave_time(self):

        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):

        return self.dic_feature

    def get_state(self, list_state_features):

        dic_state = {state_feature_name: self.dic_feature[state_feature_name]
                     for state_feature_name in list_state_features}

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
            self.dic_feature["stop_vehicle_thres1"])

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward

    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_sub_current_step[veh_id][
                get_traci_constant_mapping("VAR_LANEPOSITION")]
            speed = self.dic_vehicle_sub_current_step[veh_id][
                get_traci_constant_mapping("VAR_SPEED")]
            return pos, speed
        except:
            return None, None


class SumoEnv(AnonEnv):
    def __init__(self, dic_path, dic_traffic_env_conf):
        super().__init__(dic_path, dic_traffic_env_conf)

    def reset(self):
        file_sumocfg = self.dic_path["PATH_TO_ROADNET_FILE"]
        self.eng = traci

        if self.dic_traffic_env_conf['IF_GUI']:
            sumo_cmd = ["/usr/bin/sumo-gui", '-c', file_sumocfg,
                        "--step-length",
                        str(self.dic_traffic_env_conf["INTERVAL"])]
        else:
            sumo_cmd = ["/usr/bin/sumo", '-c', file_sumocfg,
                        "--step-length",
                        str(self.dic_traffic_env_conf["INTERVAL"])]

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

        print("start sumo")
        traci.start(sumo_cmd)

        LIST_LANE_VARIABLES_TO_SUB = [
            "LAST_STEP_VEHICLE_NUMBER",
            "LAST_STEP_VEHICLE_ID_LIST",
            "LAST_STEP_VEHICLE_HALTING_NUMBER",
            "VAR_WAITING_TIME",
        ]
        for lane in self.list_lanes:
            traci.lane.subscribe(lane, [getattr(tc, var) for var in
                                        LIST_LANE_VARIABLES_TO_SUB])

        for inter in self.list_intersection:
            inter.update_current_measurements()
        state, done = self.get_state()
        return state


if __name__ == '__main__':
    os.chdir('../')
    # Out of date. refresh please.
    print('sumo env test start...')
    from configs.config_example import *
    from configs.config_phaser import *

    create_path_dir(dic_path)
    env = SumoEnv(dic_path, dic_traffic_env)
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
