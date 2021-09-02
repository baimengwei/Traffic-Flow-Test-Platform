from multiprocessing import Process
from common.generator import Generator
from misc.utils import write_summary, create_path_dir


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
            write_summary(self.dic_path,
                          self.dic_traffic_env_conf["EPISODE_LEN"],
                          self.round_number)

        create_path_dir(self.dic_path)
        p = Process(target=test_eval,
                    args=(self.round_number,
                          self.dic_path,
                          self.dic_exp_conf,
                          self.dic_agent_conf,
                          self.dic_traffic_env_conf))
        p.start()
        p.join()
