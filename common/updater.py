from configs.config_phaser import *


class Updater:
    def __init__(self, round_number, work_dir):
        self.round_number = round_number
        self.work_dir = work_dir

        self.dic_exp_conf, self.dic_agent_conf, self.dic_traffic_env_conf, \
        self.dic_path = get_conf_file(work_dir)

        set_seed(self.dic_exp_conf["SEED"] + self.round_number)

        self.list_agent = []
        agents_infos = self.dic_traffic_env_conf["LANE_PHASE_INFOS"]
        self.agent_name = self.dic_exp_conf["MODEL_NAME"]

        for inter_name in agents_infos.keys():
            dic_traffic_env_conf = \
                copy.deepcopy(update_traffic_env_info(
                    self.dic_traffic_env_conf, inter_name))
            agent = DIC_AGENTS[self.agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=dic_traffic_env_conf,
                dic_path=self.dic_path,
                round_number=self.round_number)
            self.list_agent.append(agent)

    def load_sample(self):
        self.sample_set = []
        file_dir = os.path.join(self.dic_path["PATH_TO_WORK"], "samples")
        list_file_all = os.listdir(file_dir)
        list_file = []
        for each_file in list_file_all:
            if 'total_samples' not in each_file:
                continue
            list_file.append(each_file)
        list_file = sorted(list_file)

        for sample_file in list_file:

            sample_each = []
            file_name = os.path.join(self.dic_path["PATH_TO_WORK"],
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
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            self.sample_set[idx] = sample_each[ind_sta: ind_end]
            print("memory size after forget:", len(sample_each))

    def slice_sample(self):
        for idx, sample_each in enumerate(self.sample_set):
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"],
                              len(sample_each))
            self.sample_set[idx] = random.sample(sample_each, sample_size)
            print("memory samples number:", sample_size)

    def update_network(self):
        for sample_each, agent in zip(self.sample_set, self.list_agent):
            agent.prepare_Xs_Y(sample_each)
            agent.train_network()
            agent.save_network(
                agent.inter_name + "_round_" + str(self.round_number))

    def downsamples(self):
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
