from common.parallelizer import pipeline
from common.round_learner import RoundLearner
from multiprocessing import Process
from configs.config_phaser import *


def frapplus_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                   dic_path, round_number):
    inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
    # warn("using a fix inter_name[0]")
    inter_name = inter_names[0]
    dic_traffic_env_conf = \
        update_traffic_env_info(dic_traffic_env_conf, inter_name)
    # -------------------------------------------------------------------------
    print('round %s start...' % round_number)
    learner = RoundLearner(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                           dic_path, round_number)
    learner.learn_round()


def main(args):
    """main entrance. for frapplus
    """
    dic_exp_conf, _, dic_traffic_env_conf, _ = config_all(args)
    traffic_file_list = list(dic_traffic_env_conf[
                                 "TRAFFIC_CATEGORY"]["train_all"].keys())[:5]
    # traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021'] * 3

    traffic_file_list_surplus = copy.deepcopy(traffic_file_list)
    list_pipeline = []
    for traffic_file in traffic_file_list:
        p = Process(target=pipeline, args=(args, traffic_file, frapplus_train,))
        p.start()
        list_pipeline.append(p)
        del traffic_file_list_surplus[0]

        if len(list_pipeline) >= dic_exp_conf["PIPELINE"] or \
                len(traffic_file_list_surplus) == 0:
            for p in list_pipeline:
                p.join()
            print('a batch of pipeline is finished.')
            list_pipeline = []


if __name__ == '__main__':
    """
    """
    os.chdir('../../')
    args = parse()
    print('start execute...')
    main(args)
