from multiprocessing import Process
from common.construct_sample import ConstructSample
from common.generator import Generator
from common.updater import Updater
from configs.config_phaser import *


def generator_wrapper(round_number, dic_path, dic_exp_conf,
                      dic_agent_conf, dic_traffic_env_conf):
    generator = Generator(round_number=round_number,
                          dic_path=dic_path,
                          dic_exp_conf=dic_exp_conf,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf)
    generator.generate()


def updater_wrapper(round_number, work_dir):
    updater = Updater(
        round_number=round_number,
        work_dir=work_dir
    )
    updater.load_sample()
    updater.forget_sample()
    updater.slice_sample()
    updater.update_network()
    updater.downsamples()


def test_eval(round_number, dic_path, dic_exp_conf, dic_agent_conf,
              dic_traffic_env_conf):
    generator = Generator(round_number=round_number,
                          dic_path=dic_path,
                          dic_exp_conf=dic_exp_conf,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          test_flag=True)
    generator.generate_test()


class RoundLearner:
    """used for FRAP, FRAPPlus, DQN etc.

    """

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

        pass

    def learn_round(self):
        self.round_generate_step(generator_wrapper)
        self.round_make_samples()
        self.round_update_network(updater_wrapper)
        self.round_test_eval(test_eval)
        pass

    def round_generate_step(self, callback_func):
        process_list = []
        for generate_number in range(self.dic_exp_conf["NUM_GENERATORS"]):
            work_dir = os.path.join(self.dic_path["PATH_TO_WORK"],
                                    "samples", "round_%d" % self.round_number,
                                    "generator_%d" % generate_number)
            dic_path = update_path_work(self.dic_path, work_dir)
            create_path_dir(dic_path)
            # -----------------------------------------------------
            p = Process(target=callback_func,
                        args=(self.round_number, dic_path, self.dic_exp_conf,
                              self.dic_agent_conf, self.dic_traffic_env_conf))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

    def round_make_samples(self):
        path_to_sample = os.path.join(
            self.dic_path["PATH_TO_WORK"],
            "samples", "round_%d" % self.round_number)

        cs = ConstructSample(
            path_to_samples=path_to_sample,
            round_number=self.round_number)
        cs.make_reward()

    def round_update_network(self, callback_func):
        work_dir = self.dic_path["PATH_TO_WORK"]
        p = Process(target=callback_func,
                    args=(self.round_number,
                          work_dir))
        p.start()
        p.join()

    def round_test_eval(self, callback_func):

        path_to_log = os.path.join(self.dic_path["PATH_TO_WORK"],
                                   "test_round",
                                   "round_%d" % self.round_number)
        dic_path = update_path_work(self.dic_path, path_to_log)
        create_path_dir(dic_path)

        p = Process(target=callback_func,
                    args=(self.round_number,
                          dic_path,
                          self.dic_exp_conf,
                          self.dic_agent_conf,
                          self.dic_traffic_env_conf))
        p.start()
        p.join()
