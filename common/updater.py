import pickle
import os
import random
from configs.config_constant import DIC_AGENTS
from misc.utils import set_seed


class Updater:
    def __init__(self, round_number, dic_agent_conf, dic_exp_conf,
                 dic_traffic_env_conf, dic_path):
        self.round_number = round_number
        self.dic_agent_conf = dic_agent_conf
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        set_seed(self.dic_exp_conf["SEED"] + self.round_number)

        self.agent_name = self.dic_exp_conf["MODEL_NAME"]
        self.agent = DIC_AGENTS[self.agent_name](
            self.dic_agent_conf, self.dic_traffic_env_conf,
            self.dic_path, self.round_number)

    def load_sample(self):
        self.sample_set = []
        file_name = os.path.join(self.dic_path["PATH_TO_WORK"],
                                 "samples", "total_samples.pkl")
        sample_file = open(file_name, "rb")
        try:
            while True:
                self.sample_set += pickle.load(sample_file)
        except EOFError:
            sample_file.close()
            pass

    def forget_sample(self):
        ind_end = len(self.sample_set)
        print("memory size before forget: {0}".format(ind_end))
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        self.sample_set = self.sample_set[ind_sta: ind_end]
        print("memory size after forget:", len(self.sample_set))

    def slice_sample(self):
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"],
                          len(self.sample_set))
        self.sample_set = random.sample(self.sample_set, sample_size)
        print("memory samples number:", sample_size)

    def update_network(self):
        self.agent.prepare_Xs_Y(self.sample_set)
        self.agent.train_network()
        self.agent.save_network("round_" + str(self.round_number))
