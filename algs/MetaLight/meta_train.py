import torch

from metalearner import MetaLearner
from misc.sampler import BatchSampler
from multiprocessing import Process
from configs import config_constant
from configs.config_phaser import update_traffic_env_conf2, create_dir, \
    parse_roadnet, update_path2
import time
import copy
import os
import random
import numpy as np
import tensorflow as tf
import pickle
import shutil
from configs.config_constant_traffic import *
from configs.config_phaser import parse
import sys
from configs.config_phaser import config_all


def main(args):
    """
    main entrance. for metalight and maml
    """
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        config_all(args)

    random.seed(dic_agent_conf['SEED'])
    np.random.seed(dic_agent_conf['SEED'])
    tf.set_random_seed(dic_agent_conf['SEED'])

    if not dic_agent_conf['PRE_TRAIN']:
        # build init, return a params_init.pkl
        p = Process(target=build_init,
                    args=(copy.deepcopy(dic_agent_conf),
                          copy.deepcopy(dic_traffic_env_conf),
                          copy.deepcopy(dic_path)))
        p.start()
        p.join()
    else:
        # load init
        source = os.path.join('model', 'initial', 'common', 'params_init.pkl')
        target = os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl')
        shutil.copy(source, target)

    for batch_id in range(args.run_round):
        # meta batch size process
        process_list = []
        sample_task_traffic = random.sample(
            dic_traffic_env_conf['TRAFFIC_IN_TASKS'], args.meta_batch_size)

        if dic_traffic_env_conf["MODEL_NAME"] == "MetaLight":
            p = Process(target=metalight_train,
                        args=(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path),
                              sample_task_traffic, batch_id)
                        )
            p.start()
            p.join()
        elif dic_traffic_env_conf["MODEL_NAME"] == "FRAPPlus":
            for task in sample_task_traffic:
                p = Process(target=maml_train,
                            args=(copy.deepcopy(dic_exp_conf),
                                  copy.deepcopy(dic_agent_conf),
                                  copy.deepcopy(dic_traffic_env_conf),
                                  copy.deepcopy(dic_path),
                                  task, batch_id)
                            )
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            if True:  # FIRST_PART
                meta_step(dic_path, dic_agent_conf, batch_id)
        else:
            raise ValueError

        # update the epsilon
        decayed_epsilon = dic_agent_conf["EPSILON"] * \
                          pow(dic_agent_conf["EPSILON_DECAY"], batch_id)
        dic_agent_conf["EPSILON"] = max(
            decayed_epsilon, dic_agent_conf["MIN_EPSILON"])


def build_init(dic_agent_conf, dic_traffic_env_conf,
               dic_path, traffic_file=None):
    """
        build initial model for maml and metalight
        Arguments:
            dic_agent_conf:         configuration of agent
            dic_traffic_env_conf:   configuration of traffic environment
            dic_path:               path of source files and output files
            traffic_file: None
        Returns:
            a common weight saved as params_init.pkl
        traffic_category content in each: ctg, type, roadnet_file, flow_file
    """

    if not traffic_file:
        traffic_file = dic_traffic_env_conf["TRAFFIC_CATEGORY"]["train_all"][0]

    dic_traffic_env_conf = \
        update_traffic_env_conf2(traffic_file, dic_traffic_env_conf, dic_path)

    policy = config_constant.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path)

    create_dir(dic_path)

    params = policy.init_params()
    params_dir = os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl')
    with open(params_dir, 'wb') as f:
        pickle.dump(params, f)


def preconfig_maml_train(dic_path, dic_traffic_env_conf, task):
    """
    Args:
        dic_path:
        dic_traffic_env_conf:
        task:
    Returns:
        new path and dic_traffic_env_conf by task
    """
    dic_path.update({"PATH_TO_DATA": os.path.join(
        dic_path['PATH_TO_DATA'], task.split(".")[0])})
    dic_traffic_env_conf["ROADNET_FILE"] = \
        dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
    dic_traffic_env_conf["FLOW_FILE"] = \
        dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
    roadnet_path = os.path.join(
        dic_path['PATH_TO_DATA'], dic_traffic_env_conf[
            "traffic_category"]["traffic_info"][task][2])

    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = \
        lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(len(
        lane_phase_info["intersection_1_1"][
            "start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(
        lane_phase_info["intersection_1_1"]["phase"])
    dic_traffic_env_conf["TRAFFIC_FILE"] = task
    return dic_path, dic_traffic_env_conf


def maml_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
               dic_path, task, batch_id):
    """
    Args:
        dic_exp_conf: configuration of this experiment
        dic_agent_conf: configuration of agent
        dic_traffic_env_conf: configuration of traffic environment
        dic_path: path of source files and output files
        task: traffic files name
        batch_id: round number
    Returns:
    see also. metalearner.sample_maml(task, batch_id)
    """
    dic_path, dic_traffic_env_conf = preconfig_maml_train(
        dic_path, dic_traffic_env_conf, task)

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           list_traffic_env_conf=[dic_traffic_env_conf],
                           list_path=[dic_path],
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    policy = config_constant.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path)

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              list_traffic_env_conf=[dic_traffic_env_conf],
                              list_path=[dic_path])

    if batch_id == 0:
        params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'],
                              'params_init.pkl'), 'rb'))
        metalearner.meta_params = params
        metalearner.meta_target_params = params
    else:
        params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'],
                              'params_%d.pkl' % (batch_id - 1)), 'rb'))
        metalearner.meta_params = params

        period = dic_agent_conf['PERIOD']
        target_id = int((batch_id - 1) / period)
        metalearner.meta_target_params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'],
                              'params_%d.pkl' % (target_id * period)), 'rb'))
    metalearner.sample_maml(task, batch_id)


def preconfig_metalight_train(_dic_traffic_env_conf, _dic_path, task):
    """

    Args:
        _dic_traffic_env_conf:
        _dic_path:
        task: one traffic file from tasks

    Returns:
        new _dic_traffic_env_conf and _dic_path
    """
    dic_traffic_env_conf = copy.deepcopy(_dic_traffic_env_conf)
    dic_path = copy.deepcopy(_dic_path)
    dic_path = update_path2(task, dic_path)

    # parse roadnet
    # traffic info for each content: ctg, type, roadnet, flow
    dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf[
        "TRAFFIC_CATEGORY"]["traffic_info"][task][2]
    dic_traffic_env_conf["FLOW_FILE"] = \
        dic_traffic_env_conf["TRAFFIC_CATEGORY"]["traffic_info"][task][3]
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                dic_traffic_env_conf["ROADNET_FILE"])
    # dic_traffic_env_conf['ROADNET_FILE'])
    # # a call
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = \
        lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(len(
        lane_phase_info["intersection_1_1"][
            "start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = \
        len(lane_phase_info["intersection_1_1"]["phase"])

    dic_traffic_env_conf["TRAFFIC_FILE"] = task
    return dic_traffic_env_conf, dic_path

def metalight_train(dic_exp_origin, dic_agent_origin, _dic_traffic_env_origin,
                    _dic_path_origin, traffic_files, round_number):
    """

    Args:
        dic_exp_origin:
        dic_agent_origin:
        _dic_traffic_env_origin:
        _dic_path_origin:
        traffic_files: traffic files name(different) in this round
        round_number: round number
    Returns:

    """

    tot_path = []
    tot_traffic_env = []

    for traffic_file in traffic_files:
        dic_traffic_env_conf = copy.deepcopy(_dic_traffic_env_origin)
        dic_path = copy.deepcopy(_dic_path_origin)
        dic_traffic_env_conf = \
            update_traffic_env_conf2(traffic_file,dic_traffic_env_conf,dic_path)
        dic_path = update_path2(traffic_file, dic_path)
        tot_traffic_env.append(dic_traffic_env_conf)
        tot_path.append(dic_path)

    sampler = BatchSampler(dic_exp_conf=dic_exp_origin,
                           dic_agent_conf=dic_agent_origin,
                           list_traffic_env_conf=tot_traffic_env,
                           list_path=tot_path)

    policy = config_constant.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_origin,
        dic_traffic_env_conf=tot_traffic_env,
        dic_path=tot_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_origin,
                              list_traffic_env_conf=tot_traffic_env,
                              list_path=tot_path
                              )

    if round_number == 0:
        with open(os.path.join(dic_path['PATH_TO_MODEL'],
                               'params_init.pkl'), 'rb') as f:
            params = pickle.load(f)
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        metalearner.meta_target_params = params
    else:
        params = pickle.load(
            open(os.path.join(_dic_path['PATH_TO_MODEL'],
                              'params_%d.pkl' % (batch_id - 1)), 'rb'))
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params

        period = dic_agent_conf['PERIOD']
        target_id = int((batch_id - 1) / period)

        meta_params = pickle.load(
            open(os.path.join(_dic_path['PATH_TO_MODEL'],
                              'params_%d.pkl' % (target_id * period)), 'rb'))
        meta_params = [meta_params] * len(policy.policy_inter)
        metalearner.meta_target_params = meta_params
    # for each round:
    # after define meta_params and meta_target_params, start execute...
    metalearner.sample_metalight(tasks, batch_id)


def meta_step(dic_path, dic_agent_conf, batch_id):
    """
        update the common model's parameters of metalight
        Args:
            dic_agent_conf:     dict,   configuration of agent
            dic_path:           dict,   path of source files and output files
            batch_id:           int,    round number
    """
    # get grads from local.
    grads = []
    try:
        f = open(os.path.join(dic_path['PATH_TO_GRADIENT'],
                              "gradients_%d.pkl") % batch_id, "rb")
        while True:
            grads.append(pickle.load(f))
    except BaseException:
        pass
    # load meta params.
    if batch_id == 0:
        meta_params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'],
                              'params_init.pkl'), 'rb'))
    else:
        meta_params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'],
                              'params_%d.pkl' % (batch_id - 1)), 'rb'))
    # calculate total grads.
    tot_grads = dict(zip(meta_params.keys(), [0] * len(meta_params.keys())))
    for key in meta_params.keys():
        for g in grads:
            tot_grads[key] += g[key]
    # apply total grads to meta_params.
    _beta = dic_agent_conf['BETA']
    meta_params = dict(zip(meta_params.keys(), [
        meta_params[key] - _beta * tot_grads[key] for key in
        meta_params.keys()]))

    # save the meta parameters to local.
    pickle.dump(meta_params, open(
        os.path.join(dic_path['PATH_TO_MODEL'],
                     'params_' + str(batch_id) + ".pkl"), 'wb'))


if __name__ == '__main__':
    """
    python meta_train.py --memo ${memo} --algorithm MetaLight
    """
    import os

    os.chdir('../../')
    args = parse()
    print('start execute...')
    main(args)
