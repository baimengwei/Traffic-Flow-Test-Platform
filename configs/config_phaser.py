from configs.config_constant import *
import time
from configs import config_constant
import argparse
from misc.utils import *


def update_traffic_env_feature(dic_conf, algorithm):
    """update LIST_STATE_FEATURE.
    """
    if algorithm in RL_ALGORITHM:
        dic_conf["LIST_STATE_FEATURE"] = \
            ["cur_phase", "lane_num_vehicle"]
    elif algorithm in TRAD_ALGORITHM:
        dic_conf["LIST_STATE_FEATURE"] = [
            "cur_phase",
            "lane_num_vehicle",
            "stop_vehicle_thres1",
            "time_this_phase",
            "lane_num_vehicle_left",
        ]
    else:
        raise ValueError
    return dic_conf


def update_traffic_env_tasks(dic_conf, mode):
    """Update TRAFFIC_IN_TASKS
    """
    dic_conf["TRAFFIC_IN_TASKS"] = dic_conf["TRAFFIC_CATEGORY"][mode]
    return dic_conf


def update_traffic_env_infos(dic_conf, dic_path):
    """Update TRAFFIC_FILE, LANE_PHASE_INFOS, this function use dic_path for
    search the roadnet file. should ensure the correct config in dic_path
    """
    lane_phase_infos = parse_roadnet(dic_path["PATH_TO_ROADNET_FILE"])
    dic_conf["LANE_PHASE_INFOS"] = lane_phase_infos
    dic_conf["TRAFFIC_FILE"] = dic_path["PATH_TO_DATA"].split('/')[-1]
    return dic_conf


def update_traffic_env_info(dic_conf, inter_name):
    """Update INTER_NAME, LANE_PHASE_INFO
    """
    dic_conf["INTER_NAME"] = inter_name
    dic_conf["LANE_PHASE_INFO"] = dic_conf["LANE_PHASE_INFOS"][inter_name]
    return dic_conf


def update_path_basic(dic_conf, algorithm):
    """update PATH_TO_MODEL, PATH_TO_WORK, PATH_TO_GRADIENT,
     PATH_TO_ERROR, PATH_TO_FIGURE
    """
    _time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    _time = _time + "_No" + str(random.random()) + "_"
    inner_memo = "_" + _time + algorithm

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
    })
    return dic_conf


def update_path_file(dic_conf, traffic_file):
    """Update PATH_TO_DATA, PATH_TO_ROADNET_FILE, PATH_TO_FLOW_FILE
    """
    _, roadnet_file, flow_file = get_file_detail(traffic_file)
    dic_conf.update({
        "PATH_TO_DATA":
            os.path.join(dic_conf["PATH_TO_DATA_ROOT"], traffic_file),
        "PATH_TO_ROADNET_FILE":
            os.path.join(dic_conf["PATH_TO_DATA_ROOT"], traffic_file,
                         roadnet_file),
        "PATH_TO_FLOW_FILE":
            os.path.join(dic_conf["PATH_TO_DATA_ROOT"], traffic_file,
                         flow_file),
    })
    return dic_conf


def update_path_work(dic_conf, work_dir):
    """update PATH_TO_WORK. not used or as root please
    """
    dic_conf = copy.deepcopy(dic_conf)
    dic_conf.update(
        {
            "PATH_TO_WORK": work_dir,
        }
    )
    return dic_conf


def update_path_model(dic_conf, model_dir):
    dic_conf.update(
        {
            "PATH_TO_MODEL": model_dir,
        }
    )
    return dic_conf


def config_all(args):
    """get initial four configs dict as origin.
    """
    # -------------------------------------------------------------------------
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
        "PATH_TO_ROADNET_FILE": None,
        "PATH_TO_FLOW_FILE": None,

        # PATH_TO_WORK will change in different scenario
    }
    dic_path_origin = update_path_basic(dic_path_origin, args.algorithm)
    # dic_path_origin = update_path_file(dic_path_origin, args.traffic_file)
    # update_path_work(dic_conf, work_dir)
    # -------------------------------------------------------------------------
    dic_exp_origin = {
        "MODEL_NAME": args.algorithm,
        "TRAIN_ROUND": args.train_round,
        "TASK_ROUND": args.task_round,
        "TASK_COUNT": args.num_task,
        "ADAPT_ROUND": args.adapt_round,
        "NUM_GENERATORS": args.num_generators,
        "PIPELINE": args.pipeline,
        "EXP_DEBUG": args.exp_debug,
        "SEED": args.seed,
    }
    # -------------------------------------------------------------------------
    dic_traffic_env_origin = {
        # ---------------for engine---------------------
        "INTERVAL": 1,
        "THREADNUM": 1,
        "SAVEREPLAY": args.replay,
        "RLTRAFFICLIGHT": True,
        # ---------------for mdp process----------------
        "EPISODE_LEN": args.episode_len,
        "DONE_ENABLE": args.done,
        "REWARD_NORM": args.reward_norm,
        "DIC_FEATURE_DIM": DIC_FEATURE_DIM,
        "LIST_STATE_FEATURE_ALL": LIST_STATE_FEATURE,
        "DIC_REWARD_INFO_ALL": DIC_REWARD_INFO,
        "DIC_REWARD_INFO": {"sum_stop_vehicle_thres1": -0.25},
        "MIN_ACTION_TIME": 10,
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        # ---------------for setting--------------------
        "ENV_DEBUG": args.env_debug,
        "VALID_THRESHOLD": 30,
        "ENV_NAME": args.env,
        "TRAFFIC_CATEGORY": TRAFFIC_CATEGORY,

        "LIST_STATE_FEATURE": None,

        "TRAFFIC_FILE": None,
        "LANE_PHASE_INFOS": None,

        "INTER_NAME": None,
        "LANE_PHASE_INFO": None,

        "TRAFFIC_IN_TASKS": None,
    }
    dic_traffic_env_origin = \
        update_traffic_env_feature(dic_traffic_env_origin, args.algorithm)
    # dic_traffic_env_origin = \
    #     update_traffic_env_infos(dic_traffic_env_origin,dic_path_origin)
    # dic_traffic_env_origin = \
    #     update_traffic_env_info(dic_traffic_env_origin, 'inter_name')
    # dic_traffic_env_origin = \
    #     update_traffic_env_tasks(dic_traffic_env_origin, 'train_all')

    dic_agent_origin = \
        getattr(config_constant, "DIC_AGENT_CONF_%s" %
                format(args.algorithm.upper()))

    return dic_exp_origin, dic_agent_origin, \
           dic_traffic_env_origin, dic_path_origin


def parse():
    parser = argparse.ArgumentParser(description='RLSignal')
    # ------------------------------path.conf----------------------------------
    parser.add_argument("--memo", type=str, default="memo_name")
    # ------------------------------exp.conf-----------------------------------
    parser.add_argument("--algorithm", type=str, default="MetaLight")
    parser.add_argument("--train_round", type=int, default=200,
                        help="for train process")
    parser.add_argument("--task_round", type=int, default=20,
                        help="for metalight train process")
    parser.add_argument("--num_task", type=int, default=3,
                        help="for metalight train process")
    parser.add_argument("--adapt_round", type=int, default=50,
                        help="for metalight valid test")
    parser.add_argument("--num_generators", type=int, default=3)
    parser.add_argument("--pipeline", type=int, default=3)
    parser.add_argument("--exp_debug", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    # -----------------------------traffic_env.conf---------------------------
    parser.add_argument("--replay", action="store_true", help='for engine')
    parser.add_argument("--episode_len", type=int, default=3600, help='for mdp')
    parser.add_argument("--done", action="store_true", help='for mdp')
    parser.add_argument("--reward_norm", action="store_true", help='for mdp')
    parser.add_argument("--env_debug", action="store_true", help='for setting')
    parser.add_argument("--env", type=str, default="AnonEnv", help='for se...')
    # ------------------------------------------------------------------
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.chdir('../')
