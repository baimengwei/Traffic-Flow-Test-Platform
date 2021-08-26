from metalearner import MetaLearner
import time
import copy
from multiprocessing import Process
import pickle
import random
import numpy as np
import tensorflow as tf
import os
from copy import deepcopy


def preconfig_main(_dic_exp_conf, _dic_agent_conf, _dic_traffic_env_conf,
                   _dic_path, traffic_file, args):
    """
    Returns:
        new config content based on traffic file.
    TRAFFIC_FILE and TRAFFIC_IN_TASKS is same, but the latter is a list with
     one item.
    """
    dic_exp_conf = deepcopy(_dic_exp_conf)
    dic_agent_conf = deepcopy(_dic_agent_conf)
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    dic_path = deepcopy(_dic_path)

    traffic_of_tasks = [traffic_file]

    dic_traffic_env_conf['ROADNET_FILE'] = dic_traffic_env_conf[
        "TRAFFIC_CATEGORY"]["traffic_info"][traffic_file][2]
    dic_traffic_env_conf['FLOW_FILE'] = dic_traffic_env_conf[
        "TRAFFIC_CATEGORY"]["traffic_info"][traffic_file][3]

    _time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    inner_memo = "_" + _time + args.algorithm
    dic_path.update(
        {"PATH_TO_MODEL":
         os.path.join(dic_path["PATH_TO_MODEL"], inner_memo,
                      traffic_file + "_" + _time),
         "PATH_TO_WORK":
             os.path.join(dic_path["PATH_TO_WORK"], inner_memo,
                          traffic_file + "_" + _time),
         "PATH_TO_GRADIENT":
             os.path.join(dic_path["PATH_TO_GRADIENT"], inner_memo,
                          traffic_file + "_" + _time, "gradient"),
         "PATH_TO_DATA":
             os.path.join(dic_path["PATH_TO_DATA"],
                          traffic_file.split(".")[0])})
    # traffic env
    dic_traffic_env_conf["TRAFFIC_FILE"] = traffic_file
    dic_traffic_env_conf["TRAFFIC_IN_TASKS"] = [traffic_file]
    # parse roadnet
    roadnet_path = os.path.join(
        dic_path['PATH_TO_DATA'],
        dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(len(
        lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(
        lane_phase_info["intersection_1_1"]["phase"])
    dic_exp_conf.update({
        "TRAFFIC_FILE": traffic_file,  # Todo
        "TRAFFIC_IN_TASKS": traffic_of_tasks})
    return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path


def main(args):
    """
        Perform meta-testing for MAML, Metalight, Random, and Pretrained
        Args:
            args: generated in utils.py:parse()
    """
    _dic_exp_conf, _dic_agent_conf, _dic_traffic_env_conf, _dic_path = \
        config_all(args)
    traffic_file_list = _dic_traffic_env_conf[
        "TRAFFIC_CATEGORY"][args.traffic_group]

    single_process = args.single_process
    process_list = []
    for traffic_file in traffic_file_list:
        # get new config dict.
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            preconfig_main(_dic_exp_conf, _dic_agent_conf,
                           _dic_traffic_env_conf, _dic_path,
                           traffic_file, args)
        if single_process:
            _train(copy.deepcopy(dic_exp_conf),
                   copy.deepcopy(dic_agent_conf),
                   copy.deepcopy(dic_traffic_env_conf),
                   copy.deepcopy(dic_path))
        else:
            # one traffic file map one process.
            p = Process(target=_train,
                        args=(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path)))
            process_list.append(p)

    num_process = args.num_process
    if not single_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < num_process:
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < num_process:
                continue

            idle = check_all_workers_working(list_cur_p)
            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]
        for i in range(len(list_cur_p)):
            p = list_cur_p[i]
            p.join()


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def _train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    """
        Perform meta-testing for MAML, Metalight, Random, and Pretrained
        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            dic_traffic_env_conf:   list,   configuration of traffic environment
            dic_path:               list,   path of source files and output files
        for some reason, the list only have one item, and each process
            will map one task.
    """
    random.seed(dic_agent_conf['SEED'])
    np.random.seed(dic_agent_conf['SEED'])
    tf.set_random_seed(dic_agent_conf['SEED'])

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           list_traffic_env_conf=[dic_traffic_env_conf],
                           list_path=[dic_path],
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)
    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=[dic_traffic_env_conf],
        dic_path=[dic_path])
    # metalearner taks sampler(one item list) and policy (one item list)
    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              list_traffic_env_conf=[dic_traffic_env_conf],
                              list_path=[dic_path])

    if dic_agent_conf['PRE_TRAIN']:
        # generally, load params from local path: model/initial/common/xxx.pkl
        if not dic_agent_conf['PRE_TRAIN_MODEL_NAME'] == 'random':
            params = pickle.load(
                open(os.path.join('model', 'initial', "common",
                                  dic_agent_conf['PRE_TRAIN_MODEL_NAME']
                                  + '.pkl'), 'rb'))
            metalearner.meta_params = params
            metalearner.meta_target_params = params

    # episodes means buffer.
    episodes = None
    for batch_id in range(dic_exp_conf['NUM_ROUNDS']):
        task = dic_exp_conf['TRAFFIC_FILE']
        if dic_agent_conf['MULTI_EPISODES']:
            episodes = metalearner.sample_meta_test(
                task, batch_id, episodes)
        else:
            episodes = metalearner.sample_meta_test(task, batch_id)


if __name__ == '__main__':
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    main(args)
