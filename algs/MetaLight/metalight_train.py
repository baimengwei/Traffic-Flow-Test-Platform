import misc.summary as summary

from algs.MetaLight.metalight_learner import MetaLightLearner
from common.round_learner import RoundLearner
from multiprocessing import Process
from configs.config_phaser import *
from misc.utils import *


def metalight_adapt_conf(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                         dic_path, traffic_file):
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        get_deep_copy(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                      dic_path)
    # update path config -> file dir, model dir, work dir
    dic_path = update_path_file(dic_path, traffic_file)
    model_dir = os.path.join(dic_path["PATH_TO_MODEL"], 'adapt_round',
                             traffic_file, 'transition')
    dic_path = update_path_model(dic_path, model_dir)
    log_dir = os.path.join(dic_path["PATH_TO_WORK"], 'adapt_round',
                           traffic_file)
    dic_path = update_path_work(dic_path, log_dir)
    # update traffic config -> infos, info
    dic_traffic_env_conf = update_traffic_env_infos(dic_traffic_env_conf,
                                                    dic_path)
    inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
    # warn("using a fix inter_name[0]")
    inter_name = inter_names[0]
    dic_traffic_env_conf = update_traffic_env_info(dic_traffic_env_conf,
                                                   inter_name)
    create_path_dir(dic_path)
    copy_conf_file(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                   dic_path)
    return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path


def metalight_adapt(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                    dic_path, round_number):
    def adapt(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
              dic_path, round_number):
        learner = RoundLearner(dic_exp_conf, dic_agent_conf,
                               dic_traffic_env_conf,
                               dic_path, round_number)
        learner.learn_round()

    print('adapt round %d start..' % round_number)
    traffic_files = dic_traffic_env_conf["TRAFFIC_IN_TASKS"]
    list_proc = []
    for traffic_file in list(traffic_files.keys()):
        exp_conf, agent_conf, traffic_env_conf, path_conf = \
            metalight_adapt_conf(dic_exp_conf, dic_agent_conf,
                                 dic_traffic_env_conf, dic_path,
                                 traffic_file)
        p = Process(target=adapt, args=(exp_conf,
                                        agent_conf,
                                        traffic_env_conf,
                                        path_conf,
                                        round_number,))
        p.start()
        list_proc.append(p)
    for p in list_proc:
        p.join()


def metalight_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                    dic_path, round_number):
    print('round %s start...' % round_number)
    learner = MetaLightLearner(dic_exp_conf, dic_agent_conf,
                               dic_traffic_env_conf,
                               dic_path, round_number)
    learner.learn_round()


def main(args):
    """main entrance. for metalight
    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        pre_config_for_scenarios(args)
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
        _, _, _, path_conf = \
            metalight_adapt_conf(dic_exp_conf, dic_agent_conf,
                                 dic_traffic_env_conf,
                                 dic_path, traffic_file)
        summary.main(path_conf['PATH_TO_WORK'])
    print('plot finished.')
    # plot_msg(dic_path)
    # summary.main('records/workspace/FRAP_MMM/_08_25_13_30_44_FRAP')
    time_count = time.time() - t_start
    print('finished all. cost time: %.3f min' % (time_count / 60))


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    # args = parse()
    # main(args)

    summary.main('records/workspace/FRAP_MMM/_08_25_13_30_44_FRAP')
