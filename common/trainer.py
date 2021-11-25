import copy
import time
from multiprocessing import Process
from common.round_learner import RoundLearner
from configs.config_phaser import config_all
from misc import summary
from misc.utils import log_round_time


class Trainer:
    def __init__(self, args, traffic_file_list, *, callback=None):
        self.__args = args
        self.__traffic_file_list = traffic_file_list
        if callback is None:
            self.__callback = self.__default_train
        else:
            self.__callback = callback

    def train(self):
        """
        inputs: __traffic_file_list, __args, __callback
        outputs: __pipeline[function] call
        """
        traffic_file_list_surplus = copy.deepcopy(self.__traffic_file_list)
        list_pipeline = []
        for traffic_file in self.__traffic_file_list:
            p = Process(target=self.__pipeline,
                        args=(self.__args, traffic_file, self.__callback,))
            p.start()
            list_pipeline.append(p)
            del traffic_file_list_surplus[0]

            if len(list_pipeline) >= self.__args.num_pipeline or \
                    len(traffic_file_list_surplus) == 0:
                for p in list_pipeline:
                    p.join()
                print("join pipeline execute finished..")
                list_pipeline = []

    @staticmethod
    def __pipeline(args, traffic_file, callback_func):
        """
        """
        t_start = time.time()
        conf_exp, conf_agent, conf_traffic, conf_path = config_all(args)

        conf_path.set_traffic_file(traffic_file)
        conf_traffic.set_traffic_file(traffic_file)
        conf_path.create_path_dir()
        conf_path.dump_conf_file(conf_exp, conf_agent, conf_traffic)

        for round_number in range(args.train_round):
            t_round = time.time()
            p = Process(target=callback_func, args=(conf_path, round_number,))
            p.start()
            p.join()
            print('round %d finished..' % round_number)
            log_round_time(conf_path, round_number, t_round, time.time())

        # plot_msg(dic_path)
        summary.summary_detail_test(conf_path)
        time_count = time.time() - t_start
        print('finished %s. cost time: %.3f min' %
              (callback_func.__name__, time_count / 60))

    @staticmethod
    def __default_train(conf_path, round_number):
        print('round %s start...' % round_number)
        learner = RoundLearner(conf_path, round_number)
        learner.learn_round()
