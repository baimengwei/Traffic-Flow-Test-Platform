import os

from multiprocessing import Process
from configs.config_phaser import update_path2, create_dir
from misc.construct_sample import ConstructSample
from misc.generator import Generator
from misc.updater import Updater


class FRAPPlusLearner:
    def __init__(self, dic_exp_conf, dic_agent_conf,
                 dic_traffic_env_conf, dic_path, round_number):
        """

        """
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

    def learn_frapplus(self):
        self.round_generate_step()
        self.round_make_samples()
        self.round_update_network()
        self.round_test_eval()

    def round_generate_step(self):
        def generator_wrapper(round_number,dic_path, dic_exp_conf,
                              dic_agent_conf, dic_traffic_env_conf):
            generator = Generator(round_number=round_number,
                                  dic_path=dic_path,
                                  dic_exp_conf=dic_exp_conf,
                                  dic_agent_conf=dic_agent_conf,
                                  dic_traffic_env_conf=dic_traffic_env_conf)
            generator.generate()

        process_list = []
        for generate_number in range(self.dic_exp_conf["NUM_GENERATORS"]):
            path_to_log = os.path.join(
                self.dic_path["PATH_TO_WORK"],
                "samples", "round_%d" % self.round_number,
                           "generator_%d" % generate_number)
            dic_path = update_path2(path_to_log, self.dic_path)
            create_dir(dic_path)
            # -----------------------------------------------------
            p = Process(target=generator_wrapper,
                        args=(self.round_number, dic_path, self.dic_exp_conf,
                              self.dic_agent_conf, self.dic_traffic_env_conf))
            p.start()
            process_list.append(p)
        for i in range(len(process_list)):
            p = process_list[i]
            p.join()

    def round_make_samples(self):
        path_to_sample = os.path.join(
            self.dic_path["PATH_TO_WORK"],
            "samples", "round_%d" % self.round_number)

        cs = ConstructSample(
            path_to_samples=path_to_sample,
            round_number=self.round_number,
            dic_traffic_env_conf=self.dic_traffic_env_conf)
        cs.make_reward()

    def round_update_network(self):
        def updater_wrapper(round_number, dic_agent_conf,
                            dic_exp_conf, dic_traffic_env_conf,
                            dic_path):
            updater = Updater(
                round_number=round_number,
                dic_agent_conf=dic_agent_conf,
                dic_exp_conf=dic_exp_conf,
                dic_traffic_env_conf=dic_traffic_env_conf,
                dic_path=dic_path
            )
            updater.load_sample()
            updater.forget_sample()
            updater.slice_sample()
            updater.update_network()

        p = Process(target=updater_wrapper,
                    args=(self.round_number,
                          self.dic_agent_conf,
                          self.dic_exp_conf,
                          self.dic_traffic_env_conf,
                          self.dic_path))
        p.start()
        p.join()

    def round_test_eval(self):
        def test_eval(round_number, dic_path, dic_exp_conf, dic_agent_conf,
                      dic_traffic_env_conf):
            generator = Generator(round_number=round_number,
                                  dic_path=dic_path,
                                  dic_exp_conf=dic_exp_conf,
                                  dic_agent_conf=dic_agent_conf,
                                  dic_traffic_env_conf=dic_traffic_env_conf)
            generator.generate_test()

        path_to_log = os.path.join(self.dic_path["PATH_TO_WORK"],
                                   "test_round",
                                   "round_%d" % self.round_number)
        self.dic_path = update_path2(path_to_log, self.dic_path)
        create_dir(self.dic_path)

        p = Process(target=test_eval,
                    args=(self.round_number,
                          self.dic_path,
                          self.dic_exp_conf,
                          self.dic_agent_conf,
                          self.dic_traffic_env_conf))
        p.start()
        p.join()
