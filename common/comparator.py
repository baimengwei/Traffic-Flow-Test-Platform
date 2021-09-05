from configs.config_phaser import *


class Comparator:
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, traffic_tasks, round_number):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.traffic_tasks = traffic_tasks
        self.round_number = round_number

        agent_name = self.dic_exp_conf["MODEL_NAME"]
        self.meta_agent = DIC_AGENTS[agent_name](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            round_number=self.round_number,
            mode='meta')

    def generate_compare(self):
        self.list_samples = []
        for traffic_task in self.traffic_tasks:
            sample_set = []
            file_name = os.path.join(
                self.dic_path["PATH_TO_WORK"], "../", "task_round",
                "round_%d" % self.round_number, traffic_task, "samples",
                "total_samples.pkl")
            sample_file = open(file_name, "rb")
            try:
                while True:
                    sample_set += pickle.load(sample_file)
            except EOFError:
                sample_file.close()
                pass
            ind_end = len(sample_set)
            print("memory size before forget: {0}".format(ind_end))
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            sample_set = sample_set[ind_sta: ind_end]
            print("memory size after forget:", len(sample_set))
            self.list_samples.append(sample_set)

    def generate_target(self):
        self.list_targets = []
        for idx, traffic_task in enumerate(self.traffic_tasks):
            agent_name = self.dic_exp_conf["MODEL_NAME"]
            model_dir = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                     "../", "task_round",
                                     "round_%d" % self.round_number,
                                     traffic_task)
            dic_path = \
                update_path_model(copy.deepcopy(self.dic_path), model_dir)
            agent_task = DIC_AGENTS[agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=dic_path,
                round_number=self.dic_exp_conf["TASK_ROUND"] - 1,
                mode='task')
            sample_set = self.list_samples[idx]
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"],
                              len(sample_set))
            sample_set_target = random.sample(sample_set, sample_size)
            Xs, Y = agent_task.prepare_Xs_Y_meta(sample_set_target)
            self.list_targets += zip(Xs, Y)

    def update_meta_agent(self):
        self.meta_agent.train_network_meta(np.array(self.list_targets))

    def save_meta_agent(self):
        self.meta_agent.save_network_meta('round_%d' % self.round_number)
