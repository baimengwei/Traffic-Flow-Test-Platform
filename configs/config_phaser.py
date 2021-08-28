from collections import OrderedDict
from configs.config_constant import *
from configs.config_constant_traffic import *
import time
import os
from configs import config_constant
import argparse
import json

from misc.utils import copy_conf_file, copy_traffic_file, \
    get_relation, get_phase_map


def update_traffic_env_conf(args, dic_conf, dic_path):
    """update LIST_STATE_FEATURE, TRAFFIC_IN_TASKS, ROADNET_FILE,FLOW_FILE
    LANE_PHASE_INFOS

    Returns:
        dic_conf
    """
    if args.algorithm in RL_ALGORITHM:
        dic_conf["LIST_STATE_FEATURE"] = \
            ["cur_phase", "lane_num_vehicle"]
    elif args.algorithm in TRAD_ALGORITHM:
        dic_conf["LIST_STATE_FEATURE"] = [
            "cur_phase",
            "lane_num_vehicle",
            "stop_vehicle_thres1",
        ]
    else:
        raise ValueError

    dic_conf["TRAFFIC_IN_TASKS"] = dic_conf[
        "TRAFFIC_CATEGORY"][args.traffic_group]

    traffic_file = dic_conf["TRAFFIC_FILE"]
    dic_conf["ROADNET_FILE"] = dic_conf["TRAFFIC_CATEGORY"][
        "traffic_info"][traffic_file][2]
    dic_conf["FLOW_FILE"] = dic_conf["TRAFFIC_CATEGORY"][
        "traffic_info"][traffic_file][3]
    # parse roadnet
    roadnet_file_dir = os.path.join(dic_path['PATH_TO_DATA_ROOT'],
                                    traffic_file.split(".")[0],
                                    dic_conf["ROADNET_FILE"])
    lane_phase_infos = parse_roadnet(roadnet_file_dir)
    dic_conf["LANE_PHASE_INFOS"] = lane_phase_infos

    return dic_conf


def update_traffic_env_conf2(inter_name, dic_conf):
    """Update INTER_NAME, LANE_PHASE_INFO

    Args:
        inter_name:
        dic_conf:
    Returns:

    """
    dic_conf["INTER_NAME"] = inter_name
    dic_conf["LANE_PHASE_INFO"] = dic_conf["LANE_PHASE_INFOS"][inter_name]
    return dic_conf


def check_value(dict_conf):
    """check whether the dict value have None, None value will raise an
     exception

    """
    for key in dict_conf.keys():
        if dict_conf[key] is None:
            raise ValueError


def create_dir(dic_path):
    """create dir for further work

    """
    for path_name in dic_path.keys():
        if "ROOT" not in path_name and dic_path[path_name] is not None:
            if not os.path.exists(dic_path[path_name]):
                os.makedirs(dic_path[path_name])


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


def update_path(args, dic_conf):
    """update PATH_TO_MODEL, PATH_TO_WORK, PATH_TO_GRADIENT, PATH_TO_ERROR
    PATH_TO_FIGURE, PATH_TO_DATA

    """
    _time = time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time()))
    inner_memo = "_" + _time + args.algorithm
    dic_conf.update({
        "PATH_TO_MODEL":
            os.path.join(dic_conf["PATH_TO_MODEL_ROOT"], inner_memo),
        "PATH_TO_WORK":
            os.path.join(dic_conf["PATH_TO_WORK_ROOT"], inner_memo),
        "PATH_TO_GRADIENT":
            os.path.join(dic_conf["PATH_TO_GRADIENT_ROOT"], inner_memo),
        "PATH_TO_ERROR":
            os.path.join(dic_conf["PATH_TO_ERROR_ROOT"], inner_memo),
        "PATH_TO_FIGURE":
            os.path.join(dic_conf["PATH_TO_FIGURE_ROOT"], inner_memo),
        "PATH_TO_DATA":
            os.path.join(dic_conf["PATH_TO_DATA_ROOT"], args.traffic_file),
    })
    return dic_conf


def update_path2(log_dir, dic_conf):
    """update PATH_TO_LOG
    """
    dic_conf.update(
        {
            "PATH_TO_LOG": log_dir,
        }
    )
    return dic_conf


def update_path3(work_dir, dic_conf):
    """update PATH_TO_WORK
    """
    dic_conf.update(
        {
            "PATH_TO_WORK": work_dir,
        }
    )
    return dic_conf


def config_all(args):
    """get initial four configs dict as origin.

    """
    dic_path_origin = {
        "PATH_TO_MODEL_ROOT": "./records/weights/" + args.memo,
        "PATH_TO_WORK_ROOT": "./records/workspace/" + args.memo,
        "PATH_TO_ERROR_ROOT": "./records/errors/" + args.memo,
        "PATH_TO_GRADIENT_ROOT": "./records/gradient/" + args.memo,
        "PATH_TO_FIGURE_ROOT": "./records/figures/" + args.memo,
        "PATH_TO_DATA_ROOT": "./data/scenario/",

        "PATH_TO_MODEL": None,
        "PATH_TO_WORK": None,
        "PATH_TO_GRADIENT": None,
        "PATH_TO_ERROR": None,
        "PATH_TO_FIGURE": None,
        "PATH_TO_DATA": None,

        "PATH_TO_LOG": None,
        # PATH_TO_WORK will change in different scenario

        "PATH_TO_PRETRAIN_MODEL_ROOT": "./records/weights/default/",
        "PATH_TO_PRETRAIN_WORK_ROOT": "./records/workspace/default/",
    }
    dic_path_origin = update_path(args, dic_path_origin)
    # update_path2('log_dir', 'dic_path')
    # update_path3('work_dir', 'dic_path')

    dic_traffic_env_conf_origin = {

        "SAVEREPLAY": args.replay,
        "EPISODE_LEN": args.episode_len,
        "DONE_ENABLE": args.done,
        "REWARD_NORM": args.reward_norm,

        "ENV_DEBUG": args.debug,
        "FAST_BATCH_SIZE": args.fast_batch_size,
        'MODEL_NAME': args.algorithm,
        "ENV_NAME": args.env,
        "TRAFFIC_FILE": args.traffic_file,

        "DIC_FEATURE_DIM": DIC_FEATURE_DIM,
        "LIST_STATE_FEATURE_ALL": LIST_STATE_FEATURE,
        "DIC_REWARD_INFO_ALL": DIC_REWARD_INFO,
        "TRAFFIC_CATEGORY": TRAFFIC_CATEGORY,

        "DIC_REWARD_INFO": {"sum_stop_vehicle_thres1": -0.25},
        "VALID_THRESHOLD": 30,
        "MIN_ACTION_TIME": 10,
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        # print("MIN_ACTION_TIME should include YELLOW_TIME")
        "INTERVAL": 1,
        "THREADNUM": 1,
        "RLTRAFFICLIGHT": True,

        "LIST_STATE_FEATURE": None,
        "TRAFFIC_IN_TASKS": None,
        "ROADNET_FILE": None,
        "FLOW_FILE": None,
        "LANE_PHASE_INFOS": None,

        "INTER_NAME": None,
        "LANE_PHASE_INFO": None,
    }
    dic_traffic_env_conf_origin = \
        update_traffic_env_conf(args, dic_traffic_env_conf_origin,
                                dic_path_origin)
    # dic_traffic_env_conf_origin = \
    #     update_traffic_env_conf2('None',dic_traffic_env_conf_origin)

    dic_agent_conf_origin = {
        'NORM': args.norm,
        'NUM_UPDATES': args.num_updates,

        'ACTIVATION_FN': args.activation_fn,

        "ALPHA": args.alpha,
        "BETA": args.beta,

        "EPSILON": args.epsilon,
        "MIN_EPSILON": args.min_epsilon,

        'SAMPLE_SIZE': args.sample_size,

        "UPDATE_Q_BAR_FREQ": 5,
        "N_LAYER": 2,

    }
    alg = args.algorithm

    dic_extra = getattr(config_constant,
                        "DIC_%s_AGENT_CONF" % format(alg.upper()))
    dic_agent_conf_origin.update(dic_extra)

    dic_exp_conf_origin = {
        "EPISODE_LEN": args.episode_len,
        "TEST_EPISODE_LEN": args.test_episode_len,
        "MODEL_NAME": args.algorithm,
        "NUM_ROUNDS": args.run_round,
        "EXP_DEBUG": args.debug,
        "SEED": args.seed,
        "NUM_GENERATORS": 3,
        "NUM_EPISODE": 1,
        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 1,
        "PRETRAIN": False,
        "PRETRAIN_NUM_ROUNDS": 20,
        "PRETRAIN_NUM_GENERATORS": 15,
        "EARLY_STOP": False,
        "TRAFFIC_FILE": [None],
        "TIME_COUNTS": 3600,
        "TEST_RUN_COUNTS": None,
    }

    return dic_exp_conf_origin, dic_agent_conf_origin, \
           dic_traffic_env_conf_origin, dic_path_origin


def pre_config_for_scenarios(args, scenarios):
    """
    Args:
        args: cmd args.
        scenarios: train_round, test_round, sample_round etc.

    Returns:
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path
    Notes:
        copy files to work dir, set seed

    """
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        config_all(args)
    dir_log_root = os.path.join(dic_path['PATH_TO_WORK'],
                                scenarios)
    dic_path = update_path2(dir_log_root, dic_path)
    create_dir(dic_path)
    copy_conf_file(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                   dic_path)
    copy_traffic_file(dic_traffic_env_conf, dic_path)
    return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path


def parse():
    # phaser
    parser = argparse.ArgumentParser(description='RLSignal')
    # ---------------------------------exp.conf
    # --------------------------------
    parser.add_argument("--memo", type=str, default="memo_name")
    parser.add_argument("--algorithm", type=str, default="MetaLight")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--num_process", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument('--num_updates', type=int, default=100,
                        help='number of learning for each epoch')
    parser.add_argument('--fast_batch_size', type=int, default=3,
                        help='batch size for each individual task')
    parser.add_argument("--run_round", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--traffic_group", type=str, default="train_all")
    parser.add_argument("--seed", type=int, default=11)
    # ---------------------------traffic_env.conf
    # ------------------------------
    parser.add_argument("--min_action_time", type=int, default=10)
    parser.add_argument("--norm", type=str, default='None')
    parser.add_argument("--reward_norm", action="store_true")
    parser.add_argument("--env", type=str, default="AnonEnv")
    parser.add_argument("--episode_len", type=int, default=3600)
    parser.add_argument("--test_episode_len", type=int, default=3600)
    parser.add_argument("--done", action="store_true")
    parser.add_argument("--traffic_file", type=str,
                        default="demo_train_1364")
    # ------------------------------agent.conf
    # ---------------------------------
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--min_lr", type=float, default=0.001)
    parser.add_argument('--activation_fn', type=str, default='leaky_relu')
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.001)
    # parser.add_argument("--early_stop", action="store_true")
    # parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--min_epsilon", type=float, default=0.2)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.chdir('../')
    road_net_log = parse_roadnet('./data/template_ls/roadnet_1_1.json')
    print(road_net_log)
