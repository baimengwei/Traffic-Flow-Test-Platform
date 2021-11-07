from configs.config_phaser import *


class Updater:
    def __init__(self, conf_path: ConfPath, round_number):
        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__conf_exp, self.__conf_agent, self.__conf_traffic = self.__conf_path.load_conf_file()

        agents_infos = self.__conf_traffic.TRAFFIC_INFOS
        list_inters = sorted(list(agents_infos.keys()))
        self.__conf_path.set_work_sample_total(list_inters)
        gen_cnt = self.__conf_exp.NUM_GENERATORS
        self.__conf_path.set_work_sample_each(self.__round_number, gen_cnt, list_inters)

        agent_name = self.__conf_exp.MODEL_NAME
        agent_class = __import__('algs.%s.%s_agent.%sAgent'
                                 % (agent_name.upper(),
                                    agent_name.lower(),
                                    agent_name.title()))
        self.__list_agent = []
        for inter_name in list_inters:
            self.__conf_traffic.set_intersection(inter_name)
            agent = agent_class(self.__conf_agent,
                                self.__conf_traffic,
                                self.__conf_path)
            self.__list_agent.append(agent)

    def load_sample(self):
        self.sample_set = []
        for sample_file in self.__conf_path.WORK_SAMPLE_TOTAL:
            sample_each = []
            file_name = os.path.join(self.__conf_path.WORK,
                                     "samples", sample_file)
            sample_file = open(file_name, "rb")
            try:
                while True:
                    sample_each += pickle.load(sample_file)
            except EOFError:
                sample_file.close()
                pass
            self.sample_set.append(sample_each)

    def forget_sample(self):
        for idx, sample_each in enumerate(self.sample_set):
            ind_end = len(sample_each)
            print("memory size before forget: {0}".format(ind_end))
            ind_sta = max(0, ind_end - self.__conf_agent.MAX_MEMORY_LEN)
            self.sample_set[idx] = sample_each[ind_sta: ind_end]
            print("memory size after forget:", len(sample_each))

    def slice_sample(self):
        for idx, sample_each in enumerate(self.sample_set):
            sample_size = min(self.__conf_agent.SAMPLE_SIZE,
                              len(sample_each))
            self.sample_set[idx] = random.sample(sample_each, sample_size)
            print("memory samples number:", sample_size)

    def update_network(self):
        for sample_each, agent in zip(self.sample_set, self.__list_agent):
            agent.prepare_Xs_Y(sample_each)
            agent.train_network()
            agent.save_network(
                agent.inter_name + "_round_" + str(self.__round_number))

    def downsamples(self):
        for log_file in self.__conf_path.WORK_SAMPLE_EACH:
            downsample(log_file)
