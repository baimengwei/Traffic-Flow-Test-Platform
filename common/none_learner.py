import os
from multiprocessing import Process
from configs.config_phaser import create_dir, update_path2
from misc.generator import Generator
from misc.utils import write_summary


class NoneLearner:
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number

        pass

    def learn_round(self):
        self.round_test_eval()
        pass

    def round_test_eval(self):
        def test_eval(round_number, dic_path, dic_exp_conf, dic_agent_conf,
                      dic_traffic_env_conf):
            generator = Generator(round_number=round_number,
                                  dic_path=dic_path,
                                  dic_exp_conf=dic_exp_conf,
                                  dic_agent_conf=dic_agent_conf,
                                  dic_traffic_env_conf=dic_traffic_env_conf)
            generator.generate(done_enable=False)
            write_summary(self.dic_path, 3600, self.round_number)

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
