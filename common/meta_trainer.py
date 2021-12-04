import time
import numpy as np

from common.meta_learner import MetaLearner
from configs.config_phaser import config_all
from misc import summary
from misc.utils import log_round_time


class MetaTrainer:
    def __init__(self, args, traffic_file_list):
        self.__args = args
        self.__traffic_file_list = traffic_file_list

    def train(self):
        """
        inputs: __traffic_file_list, __args, __callback
        outputs: __pipeline[function] call
        """
        t_start = time.time()
        conf_exp, conf_agent, conf_traffic, conf_path = config_all(self.__args)
        for round_number in range(self.__args.train_round):
            t_round = time.time()
            traffic_files = np.random.choice(self.__traffic_file_list,
                                             size=conf_agent["FILE_SIZE"],
                                             replace=False)

            conf_path.create_path_dir()
            conf_path.dump_conf_file(conf_exp, conf_agent, conf_traffic)

            print('round %s start...' % round_number)
            learner = MetaLearner(conf_path, round_number, traffic_files)
            learner.learn_round()

            print('round %d finished..' % round_number)
            log_round_time(conf_path, round_number, t_round, time.time())

        # plot_msg(dic_path)
        # summary.summary_detail_test(conf_path)
        time_count = time.time() - t_start
        print('finished . cost time: %.3f min' % (time_count / 60))
