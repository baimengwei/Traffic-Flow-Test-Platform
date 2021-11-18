import misc.summary as summary

from algs.MetaLight.metalight_learner import MetaLightLearner
from multiprocessing import Process
from configs.config_phaser import *
from misc.utils import *


def metalight_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                    dic_path, round_number):
    print('metalight_train round %d start...' % round_number)
    learner = MetaLightLearner(dic_exp_conf, dic_agent_conf,
                               dic_traffic_env_conf, dic_path, round_number)
    learner.learn_round()


def metalight_adapt(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                    dic_path, round_number):
    print('metalight_adapt round %d start' % round_number)
    learner = MetaLightLearner(dic_exp_conf, dic_agent_conf,
                               dic_traffic_env_conf,
                               dic_path, round_number)
    learner.adapt_round()


def metalight_summary(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                      dic_path, traffic_file):
    learner = MetaLightLearner(dic_exp_conf, dic_agent_conf,
                               dic_traffic_env_conf,
                               dic_path, None)
    _, _, _, path_conf = learner.round_get_adapt_conf(traffic_file)
    summary.main(path_conf['PATH_TO_WORK'])


def main(args):
    """main entrance. for metalight
    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        config_all(args)
    dic_traffic_env_conf = \
        update_traffic_env_tasks(dic_traffic_env_conf, 'train_all')
    for round_number in range(dic_exp_conf["TRAIN_ROUND"]):
        t_round = time.time()
        p = Process(target=metalight_train,
                    args=(copy.deepcopy(dic_exp_conf),
                          copy.deepcopy(dic_agent_conf),
                          copy.deepcopy(dic_traffic_env_conf),
                          copy.deepcopy(dic_path),
                          round_number))
        p.start()
        p.join()
        print('train round %d finished..' % round_number)
        log_round_time(dic_path, round_number, t_round, time.time())
    # ------------------------------------------------------------------------
    dic_traffic_env_conf = \
        update_traffic_env_tasks(dic_traffic_env_conf, 'test_homogeneous')
    for round_number in range(dic_exp_conf["ADAPT_ROUND"]):
        t_round = time.time()
        p = Process(target=metalight_adapt,
                    args=(copy.deepcopy(dic_exp_conf),
                          copy.deepcopy(dic_agent_conf),
                          copy.deepcopy(dic_traffic_env_conf),
                          copy.deepcopy(dic_path),
                          round_number))
        p.start()
        p.join()
        print('adapt round %d finished..' % round_number)
        log_round_time(dic_path, round_number, t_round, time.time())
    # ------------------------------------------------------------------------
    for traffic_file in dic_traffic_env_conf["TRAFFIC_IN_TASKS"]:
        metalight_summary(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          dic_path, traffic_file)
    print('plot finished.')

    time_count = time.time() - t_start
    print('finished all. cost time: %.3f min' % (time_count / 60))


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    # args = parse()
    # main(args)

    summary.main('records/workspace/FRAP_MMM/_08_25_13_30_44_FRAP')
