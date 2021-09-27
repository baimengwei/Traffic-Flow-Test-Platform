from multiprocessing import Process
from common.generator import Generator
from configs.config_phaser import *
from misc.utils import *


def test_eval(round_number, dic_path, dic_exp_conf, dic_agent_conf,
              dic_traffic_env_conf):
    generator = Generator(round_number=round_number,
                          dic_path=dic_path,
                          dic_exp_conf=dic_exp_conf,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf)
    generator.generate(done_enable=False)
    for inter_name in dic_traffic_env_conf["LANE_PHASE_INFOS"]:
        write_summary(dic_path, round_number, inter_name)


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
        self.round_test_eval(test_eval)
        pass

    def round_test_eval(self, callback_func):
        work_dir = os.path.join(self.dic_path["PATH_TO_WORK"],
                                "test_round",
                                "round_%d" % self.round_number)
        dic_path = update_path_work(self.dic_path, work_dir)
        create_path_dir(dic_path)
        p = Process(target=callback_func,
                    args=(self.round_number,
                          dic_path,
                          self.dic_exp_conf,
                          self.dic_agent_conf,
                          self.dic_traffic_env_conf))
        p.start()
        p.join()

