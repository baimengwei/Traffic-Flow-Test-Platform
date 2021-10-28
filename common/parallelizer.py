from multiprocessing import Process
from configs.config_phaser import *
from misc import summary


def pipeline(args, traffic_file, callback_func):
    """for single intersection.

    Args:
        args:
        traffic_file:
        callback_func:

    Returns:

    """
    t_start = time.time()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
        config_all(args)
    dic_path = update_path_basic(dic_path, "None", inner_project=traffic_file)
    dic_path = update_path_file(dic_path, traffic_file)

    create_path_dir(dic_path)
    copy_conf_file(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path)
    for round_number in range(args.train_round):
        t_round = time.time()
        p = Process(target=callback_func,
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
    print('finished %s. cost time: %.3f min' %
          (callback_func.__name__, time_count / 60))

