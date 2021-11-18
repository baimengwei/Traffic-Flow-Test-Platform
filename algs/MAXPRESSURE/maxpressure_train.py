import multiprocessing
from common.none_learner import NoneLearner
from configs.config_phaser import *
from misc import summary
from misc.utils import log_round_time
from multiprocessing import pool


def maxpressure_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                      dic_path, round_number):
    # inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
    # # warn("using a fix inter_name[0]")
    # inter_name = inter_names[0]
    # dic_traffic_env_conf = \
    #     update_traffic_env_info(dic_traffic_env_conf, inter_name)
    # ---------------------------------------------------------
    print('maxpressure_train round %s start...' % round_number)
    learner = NoneLearner(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          dic_path, round_number)
    learner.learn_round()


def modify_traffic_env(dic_conf):
    """Warning: modify traffic env config
    """
    dic_conf["MIN_ACTION_TIME"] = 6
    dic_conf["YELLOW_TIME"] = 5
    return dic_conf


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class Pool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super().__init__(*args, **kwargs)


def main(args):
    """main entrance. for sotl, note that this is for search the best params
    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        config_all(args)
    traffic_file = 'cps_multi_1888'
    dic_path = update_path_file(dic_path, traffic_file)

    dic_traffic_env_conf = modify_traffic_env(dic_traffic_env_conf)
    create_path_dir(dic_path)
    copy_conf_file(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path)

    multi_pool = Pool(processes=3)
    for round_number in range(3):
        t_round = time.time()
        dic_traffic_env_conf = \
            update_traffic_env_port(dic_traffic_env_conf,
                                    str(9000 + round_number))
        multi_pool.apply_async(func=maxpressure_train,
                               args=(copy.deepcopy(dic_exp_conf),
                                     copy.deepcopy(dic_agent_conf),
                                     copy.deepcopy(dic_traffic_env_conf),
                                     copy.deepcopy(dic_path),
                                     round_number,))
        log_round_time(dic_path, round_number, t_round, time.time())

    print('start search....')
    multi_pool.close()
    multi_pool.join()
    print('finished search.')
    summary.main(dic_path['PATH_TO_WORK'])
    time_count = time.time() - t_start
    print('finished. cost time: %.3f min' % (time_count / 60))


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    main(args)
