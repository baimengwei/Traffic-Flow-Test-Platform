from warnings import warn
from common.round_learner import RoundLearner
from misc import summary
from multiprocessing import Process
from configs.config_phaser import update_traffic_env_conf2, \
    update_path2, pre_config_for_scenarios
import time
import copy
import os
from configs.config_phaser import parse
from misc.utils import log_round_time, set_seed


def fraprq_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
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
    set_seed(round_number)

    dir_log_root = os.path.join(dic_path['PATH_TO_WORK'],
                                'train_round',
                                'round_' + str(round_number))
    dic_path = update_path2(dir_log_root, dic_path)
    # -------------------------------------------------------------------------
    print('round %s start...' % round_number)
    learner = RoundLearner(dic_exp_conf, dic_agent_conf,
                           dic_traffic_env_conf, dic_path, round_number)
    learner.learn_round()


def main(args):
    """main entrance. for frapplus
    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        pre_config_for_scenarios(args, 'train_round')

    for round_number in range(args.run_round):
        t_round = time.time()
        p = Process(target=fraprq_train,
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
