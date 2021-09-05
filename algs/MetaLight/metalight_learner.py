from multiprocessing import Process
from common.comparator import Comparator
from common.round_learner import RoundLearner
from configs.config_phaser import *
from common.generator import Generator
from misc.utils import *


class MetaLightLearner:
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
        self.traffic_tasks = \
            np.random.choice(tasks_all, self.dic_exp_conf["TASK_COUNT"])

        if self.round_number == 0:
            self.round_meta_save()
            pass
        else:
            self.round_task_learn()
            self.round_meta_learn()
        pass

    def adapt_round(self):
        def adapt(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                  dic_path, round_number):
            learner = RoundLearner(dic_exp_conf, dic_agent_conf,
                                   dic_traffic_env_conf,
                                   dic_path, round_number)
            learner.learn_round()

        traffic_file_list = \
            list(self.dic_traffic_env_conf["TRAFFIC_IN_TASKS"].keys())
        traffic_file_list_surplus = copy.deepcopy(traffic_file_list)
        list_pipeline = []

        for traffic_file in traffic_file_list:
            exp_conf, agent_conf, traffic_env_conf, path_conf = \
                self.round_get_adapt_conf(traffic_file)
            p = Process(target=adapt,
                        args=(exp_conf, agent_conf, traffic_env_conf,
                              path_conf, self.round_number,))
            p.start()
            list_pipeline.append(p)
            del traffic_file_list_surplus[0]
            if len(list_pipeline) >= self.dic_exp_conf["PIPELINE"] or \
                    len(traffic_file_list_surplus) == 0:
                for p in list_pipeline:
                    p.join()
                list_pipeline = []
        pass

    def round_task_learn(self):
        def task_learn(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                       dic_path):
            for i in range(dic_exp_conf["TASK_ROUND"]):
                learner = RoundLearner(dic_exp_conf, dic_agent_conf,
                                       dic_traffic_env_conf, dic_path, i)
                learner.learn_round()
            pass

        time_start = time.time()
        list_proc = []
        for traffic_task in self.traffic_tasks:
            dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
                self.round_get_task_conf(traffic_task)
            # -------------------------------------------------------------
            print('task %s learning start...' % traffic_task)
            p = Process(target=task_learn,
                        args=(dic_exp_conf,
                              dic_agent_conf,
                              dic_traffic_env_conf,
                              dic_path))
            list_proc.append(p)
            p.start()
        for p in list_proc:
            p.join()
        print('round_task_learn finished.. cost time: %.3f'
              % ((time.time() - time_start) / 60))

    def round_meta_learn(self):
        def meta_learn(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                       dic_path, traffic_tasks, round_number):
            comparator = Comparator(dic_exp_conf, dic_agent_conf,
                                    dic_traffic_env_conf, dic_path,
                                    traffic_tasks, round_number)
            comparator.generate_compare()
            comparator.generate_target()
            comparator.update_meta_agent()
            comparator.save_meta_agent()

        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            self.round_get_meta_conf()

        p = Process(target=meta_learn,
                    args=(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          dic_path, self.traffic_tasks, self.round_number))
        p.start()
        p.join()

    def round_test_eval(self):
        def test_eval(round_number, dic_path, dic_exp_conf, dic_agent_conf,
                      dic_traffic_env_conf):
            generator = Generator(round_number=round_number,
                                  dic_path=dic_path,
                                  dic_exp_conf=dic_exp_conf,
                                  dic_agent_conf=dic_agent_conf,
                                  dic_traffic_env_conf=dic_traffic_env_conf)
            generator.generate_test()

        list_proc = []
        for traffic_task in self.traffic_tasks:
            dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
                self.round_get_task_conf(traffic_task)

            p = Process(target=test_eval,
                        args=(self.round_number,
                              dic_path,
                              dic_exp_conf,
                              dic_agent_conf,
                              dic_traffic_env_conf))
            list_proc.append(p)
            p.start()
        for p in list_proc:
            p.join()

    def round_meta_save(self):
        """
        """
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            get_deep_copy(self.dic_exp_conf, self.dic_agent_conf,
                          self.dic_traffic_env_conf, self.dic_path)
        # update path -> file dir, model dir
        x = list(dic_traffic_env_conf["TRAFFIC_IN_TASKS"].keys())[0]
        dic_path = update_path_file(dic_path, x)
        x = os.path.join(dic_path["PATH_TO_MODEL"], "meta_round")
        dic_path = update_path_model(dic_path, x)
        # update traffic env -> infos, info
        dic_traffic_env_conf = \
            update_traffic_env_infos(dic_traffic_env_conf, dic_path)
        x = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
        # warn("using a fix inter_name[0]")
        dic_traffic_env_conf = \
            update_traffic_env_info(dic_traffic_env_conf, x[0])
        create_path_dir(dic_path)
        # ---------------------------------------------------------------------
        comparator = Comparator(dic_exp_conf, dic_agent_conf,
                                dic_traffic_env_conf, dic_path,
                                self.traffic_tasks, self.round_number)
        comparator.save_meta_agent()

    def round_get_task_conf(self, traffic_task):
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            get_deep_copy(self.dic_exp_conf, self.dic_agent_conf,
                          self.dic_traffic_env_conf, self.dic_path)
        # update path -> file dir, model dir, work_dir
        dic_path = update_path_file(dic_path, traffic_task)
        model_dir = os.path.join(dic_path["PATH_TO_MODEL"], 'task_round',
                                 'round_%d' % self.round_number, traffic_task)
        dic_path = update_path_model(dic_path, model_dir)
        work_dir = os.path.join(dic_path['PATH_TO_WORK'], 'task_round',
                                'round_%d' % self.round_number, traffic_task)
        dic_path = update_path_work(dic_path, work_dir)
        # update traffic env -> infos, info
        dic_traffic_env_conf = \
            update_traffic_env_infos(dic_traffic_env_conf, dic_path)
        inter_names = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
        # warn("using a fix inter_name[0]")
        inter_name = inter_names[0]
        dic_traffic_env_conf = \
            update_traffic_env_info(dic_traffic_env_conf, inter_name)
        # copy config to work dir
        create_path_dir(dic_path)
        copy_conf_file(dic_exp_conf, dic_agent_conf,
                       dic_traffic_env_conf, dic_path)
        return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path

    def round_get_meta_conf(self):
        """file dir is used to create a agent for further load_state"""
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            get_deep_copy(self.dic_exp_conf, self.dic_agent_conf,
                          self.dic_traffic_env_conf, self.dic_path)
        # ----update path -> file dir, model dir, work_dir.
        x = list(dic_traffic_env_conf["TRAFFIC_IN_TASKS"].keys())[0]
        dic_path = update_path_file(dic_path, x)
        model_dir = os.path.join(dic_path["PATH_TO_MODEL"], 'meta_round')
        dic_path = update_path_model(dic_path, model_dir)
        work_dir = os.path.join(dic_path['PATH_TO_WORK'], 'meta_round')
        dic_path = update_path_work(dic_path, work_dir)
        # ----update traffic env -> infos, info
        dic_traffic_env_conf = \
            update_traffic_env_infos(dic_traffic_env_conf, dic_path)
        x = list(dic_traffic_env_conf["LANE_PHASE_INFOS"].keys())
        # warn("using a fix inter_name[0]")
        dic_traffic_env_conf = \
            update_traffic_env_info(dic_traffic_env_conf, x[0])
        # copy config to work dir
        create_path_dir(dic_path)
        copy_conf_file(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                       dic_path)
        return dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path

    def round_get_adapt_conf(self, traffic_file):
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = \
            get_deep_copy(self.dic_exp_conf, self.dic_agent_conf,
                          self.dic_traffic_env_conf, self.dic_path)
        # update path config -> file dir, model dir, work dir
        dic_path = update_path_file(dic_path, traffic_file)
        model_dir = os.path.join(dic_path["PATH_TO_MODEL"], 'adapt_round',
                                 traffic_file, 'transition')
        dic_path = update_path_model(dic_path, model_dir)
        log_dir = os.path.join(dic_path["PATH_TO_WORK"], 'adapt_round',
                               traffic_file)
        dic_path = update_path_work(dic_path, log_dir)
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
