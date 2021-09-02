# collect the common function
import copy
import pickle
import random
import os
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
from math import isnan
import torch
import json

from matplotlib import pyplot as plt

from configs.config_constant_traffic import TRAFFIC_CATEGORY


def get_planed_entering(flow_file, episode_len):
    # todo--check with huichu about how each vehicle is inserted, according to
    # the interval. 1s error may occur.
    list_flow = json.load(open(flow_file, "r"))
    dic_traj = {'vehicle_id': [], 'planed_enter_time': []}
    for flow_id, flow in enumerate(list_flow):
        list_ts_this_flow = []
        for step in range(
                flow["startTime"], min(
                    flow["endTime"] + 1, episode_len)):
            if step == flow["startTime"]:
                list_ts_this_flow.append(step)
            elif step - list_ts_this_flow[-1] >= flow["interval"]:
                list_ts_this_flow.append(step)

        for vec_id, ts in enumerate(list_ts_this_flow):
            dic_traj['vehicle_id'].append(
                "flow_{0}_{1}".format(flow_id, vec_id))
            dic_traj['planed_enter_time'].append(ts)
            # dic_traj["flow_{0}_{1}".format(flow_id, vec_id)] = {
            # "planed_enter_time": ts}

    df = pd.DataFrame(dic_traj)
    # df.set_index('vehicle_id')
    return df
    # return pd.DataFrame(dic_traj).transpose()


def cal_travel_time(
        df_vehicle_actual_enter_leave,
        df_vehicle_planed_enter,
        episode_len):
    df_vehicle_planed_enter.set_index('vehicle_id', inplace=True)
    df_vehicle_actual_enter_leave.set_index('vehicle_id', inplace=True)
    df_res = pd.concat(
        [df_vehicle_planed_enter, df_vehicle_actual_enter_leave], axis=1,
        sort=False)
    assert len(df_res) == len(df_vehicle_planed_enter)

    df_res["leave_time"].fillna(episode_len, inplace=True)
    df_res["travel_time"] = df_res["leave_time"] - df_res["planed_enter_time"]
    travel_time = df_res["travel_time"].mean()
    return travel_time


def get_relation(phase, roadlink):
    """
    Returns:
        a metric(1X8X7) about phases relation
    """
    relations = []
    num_phase = len(phase)
    map = roadlink
    for p1 in phase:
        zeros = [0] * (num_phase - 1)
        count = 0
        for p2 in phase:
            if p1 == p2:
                continue
            if len(set(map[p1] + map[p2])) != len(map[p1]) + len(map[p2]):
                zeros[count] = 1
            count += 1
        relations.append(zeros)
    return relations


def get_phase_map(dic_phase_lane, list_lane, list_phase):
    phase_map = {}
    for each_phase in list_phase:
        phase = [0] * len(list_lane)
        entering_lane = dic_phase_lane[each_phase]
        for lane in entering_lane:
            phase[list_lane.index(lane)] = 1
        phase_map[each_phase] = phase
    return phase_map


def log_round_time(dic_path, round_number, t1, t2):
    """log time in work dir

    Args:
        dic_path:
        round_number:
        t1: start time
        t2: end time
    Returns:
    """
    work_dir = dic_path["PATH_TO_WORK"]
    f_timing = open(os.path.join(work_dir, "timing.txt"), "a+")
    content = "round_%d: %.4f\n" % (round_number, t2 - t1)
    f_timing.write(content)
    f_timing.close()


def get_vehicle_list(dict_line_vehicle):
    """from dict_line_vehicle to get a list with all vehicle

    Args:
        dict_line_vehicle: {'line_name':{vehicle_id, vehicle_id}, }
    Returns:
        vehicle_all
    """
    vehicle_all = []
    for x in dict_line_vehicle.values():
        vehicle_all += x
    vehicle_all = np.reshape(vehicle_all, (-1,)).tolist()
    return vehicle_all


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_total_traffic_volume(traffic_file):
    traffic = traffic_file.split(".json")[0]
    vol = int(traffic.split("_")[-1])
    return vol


def copy_conf_file(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                   dic_path):
    work_dir = dic_path["PATH_TO_WORK"]
    json.dump(dic_exp_conf,
              open(os.path.join(work_dir, "exp.conf"), "w"), indent=4)
    json.dump(dic_agent_conf,
              open(os.path.join(work_dir, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf,
              open(os.path.join(work_dir, "traffic_env.conf"), "w"), indent=4)
    json.dump(dic_path,
              open(os.path.join(work_dir, "path.conf"), "w"), indent=4)


def copy_traffic_file(dic_traffic_env_conf, dic_path):
    dir_work = dic_path["PATH_TO_WORK"]
    shutil.copy(dic_path["PATH_TO_ROADNET_FILE"], dir_work)
    shutil.copy(dic_path["PATH_TO_FLOW_FILE"], dir_work)


def downsample(path_to_log_file):
    path_to_pkl = path_to_log_file
    with open(path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(path_to_pkl)
    with open(path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)


def write_summary(dic_path, cnt_round):
    work_dir = os.path.join(dic_path["PATH_TO_WORK"])
    log_dir = os.path.join(dic_path["PATH_TO_WORK"], "../",
                           "test_results.csv")

    if cnt_round == 0:
        df_col = pd.DataFrame(
            columns=("round", "duration", "vec_in", "vec_out"))
        df_col.to_csv(log_dir, mode="a", index=False)
    df_vehicle_inter_0 = pd.read_csv(
        os.path.join(work_dir, "vehicle_inter_0.csv"),
        sep=',', header=0, dtype={0: str, 1: float, 2: float},
        names=["vehicle_id", "enter_time", "leave_time"])

    vehicle_in = sum(
        [int(x) for x in (df_vehicle_inter_0["enter_time"].values > 0)])
    vehicle_out = sum(
        [1 for x in df_vehicle_inter_0["leave_time"].values if not isnan(x)])
    duration = df_vehicle_inter_0["leave_time"].values \
               - df_vehicle_inter_0["enter_time"].values
    ave_duration = np.mean([time for time in duration if not isnan(time)])
    summary = {"round": [cnt_round], "duration": [ave_duration],
               "vec_in": [vehicle_in], "vec_out": [vehicle_out]}
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(log_dir, mode="a", header=False, index=False)


def convert_dic_to_df(dic):
    list_df = []
    for key in dic:
        df = pd.Series(dic[key], name=key)
        list_df.append(df)
    return pd.DataFrame(list_df)


def seed_test():
    print('random model:')
    for i in range(10):
        print(random.random(), end=' ')
    print('')
    print('numpy model:')
    for i in range(10):
        print(np.random.random(), end=' ')
    print('')
    print('torch model:')
    for i in range(10):
        print(torch.rand(1), end=' ')
    print('')
    exit(0)


def plot_msg(dic_path):
    figure_dir = dic_path["PATH_TO_FIGURE"]
    train_round = os.path.join(dic_path["PATH_TO_WORK"], 'train_round')
    round_dir = sorted(os.listdir(train_round),
                       key=lambda x: int(x.split('_')[-1]))
    plot_reward_list = []

    for each_round in round_dir:
        each_round = os.path.join(train_round, each_round)
        batch_dir = sorted(os.listdir(each_round),
                           key=lambda x: int(x.split('_')[-1]))
        for each_batch in batch_dir:
            each_batch = os.path.join(each_round, each_batch)
            record_msg_file = os.path.join(each_batch, 'record_msg.json')
            record_msg = json.load(open(record_msg_file))
            plot_reward_list += [record_msg["inter_reward_0"]]

    plt.plot(plot_reward_list)
    plt.xlabel("round_batch")
    plt.ylabel("reward_cal")
    plt.savefig(os.path.join(figure_dir, "reward_curve.png"))
    plt.show()


def get_file_detail(traffic_file):
    """
    Args:
        traffic_file: a name
    Returns:
    roadnet_file and  flow_file name
    """
    phase = None
    roadnet_file = None
    flow_file = None
    for category in TRAFFIC_CATEGORY:
        if traffic_file in list(TRAFFIC_CATEGORY[category].keys()):
            phase = TRAFFIC_CATEGORY[category][traffic_file][0]
            roadnet_file = TRAFFIC_CATEGORY[category][traffic_file][1]
            flow_file = TRAFFIC_CATEGORY[category][traffic_file][2]
    return phase, roadnet_file, flow_file


def create_path_dir(dic_path):
    """create dir for further work
    """
    for path_name in dic_path.keys():
        if "ROOT" not in path_name and dic_path[path_name] is not None:
            if not os.path.exists(dic_path[path_name]):
                os.makedirs(dic_path[path_name])


def check_value_conf(dict_conf):
    """check whether the dict value have None, None value will raise an
     exception
    """
    for key in dict_conf.keys():
        if dict_conf[key] is None:
            raise ValueError('k: %s, v: %s' % (key, dict_conf[key]))


def parse_roadnet(roadnet_file_dir):
    """
    Args:
        roadnet_file_dir: a full dir of the roadnet file.
    Returns:
        file infos
    """
    with open(roadnet_file_dir) as f:
        roadnet = json.load(f)

    intersections = [inter
                     for inter in roadnet["intersections"]
                     if not inter["virtual"]]

    lane_phase_info_dict = OrderedDict()
    for intersection in intersections:
        lane_phase_info_dict[intersection['id']] = \
            {"start_lane": [],
             "same_start_lane": [],
             "end_lane": [],
             "phase": [],
             "yellow_phase": None,
             "phase_startLane_mapping": {},
             "phase_noRightStartLane_mapping": {},
             "phase_sameStartLane_mapping": {},
             "phase_roadLink_mapping": {}}
        road_links = intersection["roadLinks"]
        start_lane = []
        same_start_lane = []
        end_lane = []
        roadlink_lane_pair = {idx: [] for idx in range(len(road_links))}
        roadlink_same_start_lane = {idx: [] for idx in range(len(road_links))}
        for idx in range(len(road_links)):
            road_link = road_links[idx]
            tmp_same_start_lane = []
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + \
                     str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + \
                     str(lane_link["endLaneIndex"])
                type = road_link['type']
                start_lane.append(sl)
                tmp_same_start_lane.append(sl)
                end_lane.append(el)
                roadlink_lane_pair[idx].append((sl, el, type))
            tmp_same_start_lane = tuple(set(tmp_same_start_lane))
            roadlink_same_start_lane[idx].append(tmp_same_start_lane)
            same_start_lane.append(tmp_same_start_lane)

        lane_phase_info_dict[intersection['id']
        ]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']
        ]["end_lane"] = sorted(list(set(end_lane)))
        lane_phase_info_dict[intersection['id']
        ]["same_start_lane"] = sorted(list(set(same_start_lane)))

        for phase_i in range(len(intersection["trafficLight"]["lightphases"])):
            if len(intersection["trafficLight"]["lightphases"]
                   [phase_i]["availableRoadLinks"]) == 0:
                lane_phase_info_dict[intersection['id']
                ]["yellow_phase"] = phase_i
                continue
            p = intersection["trafficLight"]["lightphases"][phase_i]
            lane_pair = []
            start_lane = []
            same_start_lane = []
            no_right_start_lane = []
            # get no right roadlink, start_lane list, no_right_start_lane list
            for ri in p["availableRoadLinks"]:
                for i in range(len(roadlink_lane_pair[ri])):
                    # roadlink_lane_pair each content: start_idx, end_idx, type
                    if roadlink_lane_pair[ri][i][0] not in start_lane:
                        start_lane.append(roadlink_lane_pair[ri][i][0])
                    if roadlink_lane_pair[ri][i][0] not in no_right_start_lane \
                            and roadlink_lane_pair[ri][i][2] != "turn_right":
                        no_right_start_lane.append(
                            roadlink_lane_pair[ri][i][0])
                    if roadlink_lane_pair[ri][i][2] != "turn_right":
                        lane_pair.extend(
                            roadlink_lane_pair[ri])  # no right roadlink
                if roadlink_same_start_lane[ri][0] not in same_start_lane:
                    same_start_lane.append(roadlink_same_start_lane[ri][0])
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)
            lane_phase_info_dict[intersection['id']][
                "phase_startLane_mapping"][phase_i] = start_lane
            lane_phase_info_dict[intersection['id']][
                "phase_noRightStartLane_mapping"][phase_i] = no_right_start_lane
            lane_phase_info_dict[intersection['id']][
                "phase_sameStartLane_mapping"][phase_i] = same_start_lane
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"][
                phase_i] = list(set(lane_pair))  # tmp to remove repeated

        lane_phase_info_dict[intersection['id']]["relation"] = get_relation(
            lane_phase_info_dict[intersection['id']]["phase"],
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"]
        )
        lane_phase_info_dict[intersection['id']]["phase_map"] = get_phase_map(
            lane_phase_info_dict[intersection['id']]['phase_startLane_mapping'],
            lane_phase_info_dict[intersection['id']]['start_lane'],
            lane_phase_info_dict[intersection['id']]['phase']
        )
    return lane_phase_info_dict


def get_deep_copy(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    dic_exp_conf = copy.deepcopy(dic_exp_conf)
    dic_agent_conf = copy.deepcopy(dic_agent_conf)
    dic_traffic_env_conf = copy.deepcopy(dic_traffic_env_conf)
    dic_path = copy.deepcopy(dic_path)
    return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path
