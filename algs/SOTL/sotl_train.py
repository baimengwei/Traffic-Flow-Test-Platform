import multiprocessing
from common.none_learner import NoneLearner
from configs.config_phaser import *
from misc.utils import log_round_time


def sotl_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
               dic_path, round_number):
    inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
    # warn("using a fix inter_name[0]")
    inter_name = inter_names[0]
    dic_traffic_env_conf = \
        update_traffic_env_info(dic_traffic_env_conf, inter_name)

    # ---------------------------------------------------------
    print('round %s start...' % round_number)
    learner = NoneLearner(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          dic_path, round_number)
    learner.learn_round()


def modify_traffic_env(dic_conf):
    """Warning: modify traffic env config
    """
    dic_conf["MIN_ACTION_TIME"] = 6
    dic_conf["YELLOW_TIME"] = 5
    return dic_conf


def modify_agent_conf(dic_conf, phi_min, theta, mu):
    # Warning: modify agent conf config
    dic_conf["PHI_MIN"] = phi_min
    dic_conf["THETA"] = theta
    dic_conf["MU"] = mu
    return dic_conf


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def main(args):
    """main entrance. for sotl, note that this is for search the best params
    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        config_all(args)
    dic_traffic_env_conf = modify_traffic_env(dic_traffic_env_conf)

    tmp_phi_min = [i for i in range(30)]  # 2
    tmp_theta = [i for i in range(30)]  # 4
    tmp_mu = [i for i in range(30)]  # 28

    mult_pool = Pool(processes=200)
    for round_number in range(int(30 * 30 * 30)):
        t_round = time.time()
        dic_agent_conf = \
            modify_agent_conf(dic_agent_conf,
                              tmp_phi_min[round_number % 30],
                              tmp_theta[int(round_number / 30) % 30],
                              tmp_mu[int(round_number / 900) % 30])
        mult_pool.apply_async(func=sotl_train,
                              args=(copy.deepcopy(dic_exp_conf),
                                    copy.deepcopy(dic_agent_conf),
                                    copy.deepcopy(dic_traffic_env_conf),
                                    copy.deepcopy(dic_path),
                                    round_number,))
        log_round_time(dic_path, round_number, t_round, time.time())

    print('start search....')
    mult_pool.close()
    mult_pool.join()
    print('finished search.')
    # summary.main(dic_path['PATH_TO_WORK'])
    time_count = time.time() - t_start
    print('finished. cost time: %.3f min' % (time_count / 60))


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    main(args)
