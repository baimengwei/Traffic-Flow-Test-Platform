import pickle
import random

from configs.config_phaser import *
from misc.utils import downsample


class Updater:
    def __init__(self, conf_path: ConfPath, round_number):
        self.conf_path = conf_path
        self.round_number = round_number

        list_inters = self.conf_path.load_conf_inters()
        self.conf_exp, self.conf_agent, self.conf_traffic = \
            self.conf_path.load_conf_file(inter_name=list_inters[0])

        self.conf_path.set_work_sample_total(list_inters)
        gen_cnt = self.conf_exp.NUM_GENERATORS
        self.conf_path.set_work_sample_each(
            self.round_number, gen_cnt, list_inters)

        agent_name = self.conf_exp.MODEL_NAME
        agent_package = __import__('algs.%s.%s_agent'
                                   % (agent_name.upper(),
                                      agent_name.lower()))
        agent_package = getattr(agent_package, '%s' % agent_name.upper())
        agent_package = getattr(agent_package, '%s_agent' % agent_name.lower())
        agent_class = getattr(agent_package, '%sAgent' % agent_name.upper())

        self.list_agent = []
        for inter_name in list_inters:
            # create agent
            agent = agent_class(self.conf_path,
                                self.round_number,
                                inter_name)
            self.list_agent.append(agent)

    def load_sample(self):
        self.sample_set = []
        # x = [1 for s in self.sample_set[1] if len(s[0]['cur_phase_index']) != 24]
        for sample_file in self.conf_path.WORK_SAMPLE_TOTAL:
            sample_each = []
            f = open(sample_file, "rb")
            try:
                while True:
                    sample_each += pickle.load(f)
            except EOFError:
                f.close()
                pass
            self.sample_set.append(sample_each)

    def forget_sample(self):
        for idx, sample_each in enumerate(self.sample_set):
            ind_end = len(sample_each)
            print("memory size before forget: {0}".format(ind_end))
            ind_sta = max(0, ind_end - self.conf_agent['MAX_MEMORY_LEN'])
            self.sample_set[idx] = sample_each[ind_sta: ind_end]
            print("memory size after forget:", len(sample_each))

    def slice_sample(self):
        for idx, sample_each in enumerate(self.sample_set):
            sample_size = min(self.conf_agent['SAMPLE_SIZE'],
                              len(sample_each))
            self.sample_set[idx] = random.sample(sample_each, sample_size)
            print("memory samples number:", sample_size)

    def update_network(self):
        for sample_each, agent in zip(self.sample_set, self.list_agent):
            agent.prepare_Xs_Y(sample_each)
            agent.train_network()
            agent.save_network(self.round_number)

    def downsamples(self):
        for log_file in self.conf_path.WORK_SAMPLE_EACH:
            downsample(log_file)
