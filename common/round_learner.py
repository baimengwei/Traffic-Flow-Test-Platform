from multiprocessing import Process
from misc.construct_sample import ConstructSample
from common.generator import Generator
from common.updater import Updater
from configs.config_phaser import *


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
        self.round_generate_step()
        self.round_make_samples()
        self.round_update_network()
        self.round_test_eval()
        pass

    def round_generate_step(self):
        def generator_wrapper(round_number, dic_path, dic_exp_conf,
                              dic_agent_conf, dic_traffic_env_conf):
            generator = Generator(round_number=round_number,
                                  dic_path=dic_path,
                                  dic_exp_conf=dic_exp_conf,
                                  dic_agent_conf=dic_agent_conf,
                                  dic_traffic_env_conf=dic_traffic_env_conf)
            generator.generate()

        process_list = []

        for generate_number in range(self.dic_exp_conf["NUM_GENERATORS"]):
            work_dir = os.path.join(self.dic_path["PATH_TO_WORK"],
                                    "samples", "round_%d" % self.round_number,
                                    "generator_%d" % generate_number)
            dic_path = update_path_work(self.dic_path, work_dir)
            create_path_dir(dic_path)
            # -----------------------------------------------------
            p = Process(target=generator_wrapper,
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

        for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
            for inter_name in sorted(
                    self.dic_traffic_env_conf["LANE_PHASE_INFOS"].keys()):
                path_to_log_file = os.path.join(
                    self.dic_path["PATH_TO_WORK"],
                    "samples",
                    "round_" + str(self.round_number),
                    "generator_" + str(cnt_gen),
                    "%s.pkl" % inter_name
                )
                downsample(path_to_log_file)

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
        dic_path = update_path_work(self.dic_path, path_to_log)
        create_path_dir(dic_path)

        p = Process(target=test_eval,
                    args=(self.round_number,
                          dic_path,
                          self.dic_exp_conf,
                          self.dic_agent_conf,
                          self.dic_traffic_env_conf))
        p.start()
        p.join()
