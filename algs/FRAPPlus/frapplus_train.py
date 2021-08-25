import json
from warnings import warn
import matplotlib.pyplot as plt
from algs.FRAPPlus.frapplus_learner import FRAPPlusLearner
from misc import summary
from multiprocessing import Process
from configs.config_phaser import update_traffic_env_conf2, \
    update_path2, pre_config_for_scenarios
import time
import copy
import os
from configs.config_phaser import parse
from misc.utils import log_round_time


def frapplus_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                   dic_path, round_number):
    """
    Args:
        dic_exp_conf: configuration of this experiment
        dic_agent_conf: configuration of agent
        dic_traffic_env_conf: configuration of traffic environment
        dic_path: path of source files and output files
        round_number: round number
    Returns:

    """

    inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
    warn("using a fix inter_name[0]")
    inter_name = inter_names[0]
    dic_traffic_env_conf = \
        update_traffic_env_conf2(inter_name, dic_traffic_env_conf)
    dir_log_root = os.path.join(dic_path['PATH_TO_WORK'],
                                'train_round',
                                'round_' + str(round_number))
    dic_path = update_path2(dir_log_root, dic_path)
    # -------------------------------------------------------------------------
    print('round %s start...' % round_number)
    learner = FRAPPlusLearner(dic_exp_conf, dic_agent_conf,
                              dic_traffic_env_conf, dic_path, round_number)
    learner.learn_frapplus()


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


def main(args):
    """main entrance. for frapplus
    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        pre_config_for_scenarios(args, 'train_round')

    for round_number in range(args.run_round):
        t_round = time.time()
        p = Process(target=frapplus_train,
                    args=(copy.deepcopy(dic_exp_conf),
                          copy.deepcopy(dic_agent_conf),
                          copy.deepcopy(dic_traffic_env_conf),
                          copy.deepcopy(dic_path),
                          round_number))
        p.start()
        p.join()
        print('round %d finished..' % round_number)
        log_round_time(dic_path, round_number, t_round, time.time())

    # plot_msg(dic_path)
    summary.main(dic_path['PATH_TO_WORK'])
    time_count = time.time() - t_start
    print('finished frapplus. cost time: %.3f min' % (time_count / 60))


if __name__ == '__main__':
    """
    """
    os.chdir('../../')
    args = parse()
    print('start execute...')
    main(args)
