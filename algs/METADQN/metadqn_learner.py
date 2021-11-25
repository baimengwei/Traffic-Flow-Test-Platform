from multiprocessing import Process
from common.construct_sample import ConstructSample
from common.generator import Generator
from common.updater import Updater
from misc.utils import *


class UpdaterCustom(Updater):
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number, traffic_tasks):
        self.round_number = round_number
        self.dic_agent_conf = dic_agent_conf
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.traffic_tasks = traffic_tasks

        self.agent_name = self.dic_exp_conf["MODEL_NAME"]
        self.agent = DIC_AGENTS[self.agent_name](
            self.dic_agent_conf, self.dic_traffic_env_conf,
            self.dic_path, self.round_number, self.traffic_tasks)

    def update_meta(self):
        file_name_origin = self.dic_traffic_env_conf["TRAFFIC_FILE"]
        for task in self.traffic_tasks:
            # -------------------------load sample-----------------------------
            self.sample_set = []
            file_name = os.path.join(self.dic_path["PATH_TO_WORK"],
                                     "../", "total_samples.pkl")
            file_name = file_name.replace(file_name_origin, task)

            sample_file = open(file_name, "rb")
            try:
                while True:
                    self.sample_set += pickle.load(sample_file)
            except EOFError:
                sample_file.close()
                pass
            self.forget_sample()
            self.slice_sample()
            # ------------------------update network---------------------------
            self.agent.prepare_Xs_Y(self.sample_set)
            self.agent.train_network()
        self.agent.save_network("round_" + str(self.round_number))


class GeneratorCustom(Generator):
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number, traffic_tasks):
        self.round_number = round_number
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.traffic_tasks = traffic_tasks

        self.agent_name = self.dic_exp_conf["MODEL_NAME"]
        self.agent = DIC_AGENTS[self.agent_name](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            round_number=self.round_number,
            traffic_tasks=self.traffic_tasks)

        self.env_name = self.dic_traffic_env_conf["ENV_NAME"]
        self.env = DIC_ENVS[self.env_name](self.dic_path,
                                           self.dic_traffic_env_conf)

    def save_meta_agent(self):
        self.agent.save_network('round_%d' % self.round_number)


def updater_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                    dic_path, round_number, traffic_tasks):
    updater = UpdaterCustom(
        round_number=round_number,
        dic_agent_conf=dic_agent_conf,
        dic_exp_conf=dic_exp_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
        traffic_tasks=traffic_tasks
    )
    updater.update_meta()


def generator_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                      dic_path, round_number, traffic_tasks):
    generator = GeneratorCustom(dic_exp_conf=dic_exp_conf,
                                dic_agent_conf=dic_agent_conf,
                                dic_traffic_env_conf=dic_traffic_env_conf,
                                dic_path=dic_path,
                                round_number=round_number,
                                traffic_tasks=traffic_tasks)
    generator.generate()


class METADQNLearner:
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.round_number = round_number
        pass

    def learn_round(self):
        tasks_all = list(self.dic_traffic_env_conf["TRAFFIC_IN_TASKS"].keys())
        # used in round_meta_learn
        self.traffic_tasks = \
            np.random.choice(tasks_all, self.dic_exp_conf["TASK_COUNT"])
        if self.round_number == 0:
            self.round_task_load()
            self.round_meta_save()
        else:
            self.round_generate_meta(generator_wrapper)
            self.round_make_samples()
            self.round_update_network(updater_wrapper)
            self.round_test_eval()
        pass

    def round_generate_meta(self, callback_func):
        process_list = []
        for task in self.traffic_tasks:
            for generate_number in range(self.dic_exp_conf["NUM_GENERATORS"]):
                dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
                    self.round_get_task_conf(task)
                work_dir = os.path.join(dic_path["PATH_TO_WORK"],
                                        "generator_%d" % generate_number)
                dic_path = update_path_work(dic_path, work_dir)

                create_path_dir(dic_path)
                # -----------------------------------------------------
                p = Process(target=callback_func,
                            args=(dic_exp_conf, dic_agent_conf,
                                  dic_traffic_env_conf, dic_path,
                                  self.round_number, self.traffic_tasks))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()

    def round_make_samples(self):
        for task in self.traffic_tasks:
            _, _, dic_traffic_env_conf, dic_path = \
                self.round_get_task_conf(task)
            cs = ConstructSample(
                path_to_samples=dic_path["PATH_TO_WORK"],
                round_number=self.round_number,
                dic_traffic_env_conf=dic_traffic_env_conf)
            cs.make_reward()

    def round_update_network(self, callback_func):
        task = np.random.choice(self.traffic_tasks)

        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            self.round_get_task_conf(task)
        p = Process(target=callback_func,
                    args=(dic_exp_conf,
                          dic_agent_conf,
                          dic_traffic_env_conf,
                          dic_path,
                          self.round_number,
                          self.traffic_tasks,))
        p.start()
        p.join()

    def round_test_eval(self):
        def test_eval(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                      dic_path, round_number, traffic_tasks):
            generator = GeneratorCustom(
                dic_exp_conf=dic_exp_conf,
                dic_agent_conf=dic_agent_conf,
                dic_traffic_env_conf=dic_traffic_env_conf,
                dic_path=dic_path,
                round_number=round_number,
                traffic_tasks=traffic_tasks
            )
            generator.generate_test()
        # --------------------------------------------------------------------
        for task in self.traffic_tasks:
            dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
                self.round_get_test_conf(task)
            p = Process(target=test_eval,
                        args=(dic_exp_conf,
                              dic_agent_conf,
                              dic_traffic_env_conf,
                              dic_path,
                              self.round_number,
                              self.traffic_tasks,))
            p.start()
            p.join()

    def round_task_load(self):
        """
        """
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            get_deep_copy(self.dic_exp_conf, self.dic_agent_conf,
                          self.dic_traffic_env_conf, self.dic_path)
        print('load task pretrained from the parent dir...')
        source_dir = os.path.join(dic_path["PATH_TO_MODEL"], '../', 'META_P')
        target_dir = os.path.join(dic_path["PATH_TO_MODEL"], "tasks_param")
        try:
            copy_files_best(source_dir, target_dir)
        except:
            raise FileNotFoundError("check file path: %s" % source_dir)

    def round_meta_save(self):
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            self.round_get_task_conf()
        generator = GeneratorCustom(dic_exp_conf, dic_agent_conf,
                                    dic_traffic_env_conf, dic_path,
                                    self.round_number, self.traffic_tasks)
        generator.save_meta_agent()

    def round_get_task_conf(self, traffic_file=None):
        if traffic_file is None:
            traffic_file = \
                list(self.dic_traffic_env_conf["TRAFFIC_IN_TASKS"].keys())[0]
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            get_deep_copy(self.dic_exp_conf, self.dic_agent_conf,
                          self.dic_traffic_env_conf, self.dic_path)
        # update path config -> file dir, model dir, work dir
        dic_path = update_path_file(dic_path, traffic_file)
        model_dir = os.path.join(dic_path["PATH_TO_MODEL"], 'meta_round')
        dic_path = update_path_model(dic_path, model_dir)
        work_dir = os.path.join(dic_path["PATH_TO_WORK"], 'samples',
                                'round_%d' % self.round_number, traffic_file)
        dic_path = update_path_work(dic_path, work_dir)
        # update traffic config -> infos, info
        dic_traffic_env_conf = update_traffic_env_infos(dic_traffic_env_conf,
                                                        dic_path)
        inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
        # warn("using a fix inter_name[0]")
        inter_name = inter_names[0]
        dic_traffic_env_conf = update_traffic_env_info(dic_traffic_env_conf,
                                                       inter_name)
        create_path_dir(dic_path)
        copy_conf_file(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                       dic_path)
        return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path

    def round_get_test_conf(self, traffic_file):
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            get_deep_copy(self.dic_exp_conf, self.dic_agent_conf,
                          self.dic_traffic_env_conf, self.dic_path)
        # update path config -> file, model dir, work dir
        dic_path = update_path_file(dic_path, traffic_file)
        model_dir = os.path.join(dic_path["PATH_TO_MODEL"], 'meta_round')
        dic_path = update_path_model(dic_path, model_dir)
        work_dir = os.path.join(dic_path["PATH_TO_WORK"], 'test_round',
                                'round_%d' % self.round_number, traffic_file)
        dic_path = update_path_work(dic_path, work_dir)
        # update traffic config -> infos, info
        dic_traffic_env_conf = update_traffic_env_infos(dic_traffic_env_conf,
                                                        dic_path)
        inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
        # warn("using a fix inter_name[0]")
        inter_name = inter_names[0]
        dic_traffic_env_conf = update_traffic_env_info(dic_traffic_env_conf,
                                                       inter_name)
        create_path_dir(dic_path)
        return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path
