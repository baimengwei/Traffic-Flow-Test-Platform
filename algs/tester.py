import copy
import time
from multiprocessing import Process
import multiprocessing
from multiprocessing.pool import Pool
from common.none_learner import NoneLearner
from configs.config_phaser import config_all
from misc import summary


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class CustomPool(Pool):
    Process = NoDaemonProcess


class Tester:
    def __init__(self, args, traffic_file_list, *, callback=None):
        self.__args = args
        self.__traffic_file_list = traffic_file_list
        if len(self.__traffic_file_list) > 1:
            raise ValueError('not support multi file! ')

        if callback is None:
            self.__callback = self.default_test

    def test(self):
        """
        inputs: __traffic_file_list, __args, __callback
        outputs: __pipeline[function] call
        """
        traffic_file = self.__traffic_file_list[0]
        self.__pipeline(self.__args, traffic_file, self.__callback)
        print('execute finished!')

    @staticmethod
    def __pipeline(args, traffic_file, callback_func):
        """
        """
        t_start = time.time()
        conf_exp, conf_agent, conf_traffic, conf_path = config_all(args)

        conf_path.set_traffic_file(traffic_file)
        conf_path.create_path_dir()
        conf_path.dump_conf_file(conf_exp, conf_agent, conf_traffic)
        mult_pool = CustomPool(processes=3)
        for round_number in range(3):
            mult_pool.apply_async(func=callback_func,
                                  args=(conf_path, round_number,))

        print('start search....')
        mult_pool.close()
        mult_pool.join()
        print('finished .')

        # plot_msg(dic_path)
        summary.summary_detail_test(conf_path)
        time_count = time.time() - t_start
        print('finished. cost time: %.3f min' % (time_count / 60))

    @staticmethod
    def default_test(conf_path, round_number):
        print('round %s start...' % round_number)
        learner = NoneLearner(conf_path, round_number)
        learner.test_round()
