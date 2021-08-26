import copy
from warnings import warn
import misc.summary as summary
import os
import time
from algs.FRAP.frap_learner import FRAPLearner
from multiprocessing import Process
from configs.config_phaser import \
    parse, \
    pre_config_for_scenarios, update_traffic_env_conf2, update_path2
from misc.utils import log_round_time


def frap_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
               dic_path, round_number):
    inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
    warn("using a fix inter_name[0]")
    inter_name = inter_names[0]
    dic_traffic_env_conf = \
        update_traffic_env_conf2(inter_name, dic_traffic_env_conf)

    dir_log_root = os.path.join(dic_path['PATH_TO_LOG'],
                                'round_' + str(round_number))
    dic_path = update_path2(dir_log_root, dic_path)
    # ---------------------------------------------------------
    print('round %s start...' % round_number)
    learner = FRAPLearner(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          dic_path, round_number)
    learner.learn_frap()


def main(args):
    """main entrance. for frap
    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        pre_config_for_scenarios(args, 'train_round')

    for round_number in range(args.run_round):
        t_round = time.time()
        p = Process(target=frap_train,
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
    # summary.main('records/workspace/FRAP_MMM/_08_25_13_30_44_FRAP')
    time_count = time.time() - t_start
    print('finished. cost time: %.3f min' % (time_count / 60))


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    main(args)
