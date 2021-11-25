import time

import misc.summary as summary
from multiprocessing import Process
from algs.METADQN.metadqn_learner import METADQNLearner
from common.round_learner import RoundLearner
from common.trainer import Trainer
from configs.config_phaser import *
from misc.utils import *


def metadqn_adapt(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                  dic_path, round_number):
    inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
    # warn("using a fix inter_name[0]")
    inter_name = inter_names[0]
    dic_traffic_env_conf = \
        update_traffic_env_info(dic_traffic_env_conf, inter_name)
    # ---------------------------------------------------------
    print('round %s start...' % round_number)
    learner = RoundLearner(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                           dic_path, round_number)
    learner.learn_round()


def metadqn_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                  dic_path, round_number):
    print('metalight_train round %d start...' % round_number)
    learner = MetaDQNLearner(dic_exp_conf, dic_agent_conf,
                             dic_traffic_env_conf, dic_path, round_number)
    learner.learn_round()


def metadqn_summary(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                    dic_path, traffic_file):
    learner = MetaDQNLearner(dic_exp_conf, dic_agent_conf,
                             dic_traffic_env_conf,
                             dic_path, None)
    _, _, _, path_conf = learner.round_get_task_conf(traffic_file)
    summary.main(path_conf['PATH_TO_WORK'])


def main_train(args):
    """main entrance. for metalight
    """
    conf_exp, _, conf_traffic, _ = config_all(args)
    traffic_file_list = conf_traffic.TRAFFIC_CATEGORY['train_all']
    print('training list:', traffic_file_list)

    trainer = Trainer(args, traffic_file_list)
    trainer.train()


def main_adapt(args):
    pass


def main_test(args):
    t_start = time.time()
    dic_exp_conf, _, dic_traffic_env_conf, _ = \
        config_all(args)
    list_traffic_file = list(
        dic_traffic_env_conf["TRAFFIC_CATEGORY"]["test_homogeneous"].keys())
    list_traffic_file_surplus = copy.deepcopy(list_traffic_file)
    list_pipeline = []
    # input('input your meta params file and place it(e.g. round_99.pt):')
    # ------------------------------------------------------------------------
    for traffic_file in list_traffic_file:
        p = Process(target=pipeline, args=(args, traffic_file, metadqn_adapt,))
        p.start()
        list_pipeline.append(p)
        del list_traffic_file_surplus[0]
        if len(list_pipeline) >= dic_exp_conf["PIPELINE"] or \
                len(list_traffic_file_surplus) == 0:
            for p in list_pipeline:
                p.join()
            print("join pipeline execute finished..")
            list_pipeline = []

    time_count = time.time() - t_start
    print('finished all. cost time: %.3f min' % (time_count / 60))


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    # args = parse()
    # main(args)
