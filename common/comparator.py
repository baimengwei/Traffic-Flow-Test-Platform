from multiprocessing import Process

from configs.config_phaser import *
import numpy as np

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
            traffic_tasks=self.traffic_tasks,
            round_number=self.round_number)

        self.env_name = self.dic_traffic_env_conf["ENV_NAME"]
        self.env = DIC_ENVS[self.env_name](self.dic_path,
                                           self.dic_traffic_env_conf)

    def generate_compare(self, callback_func, done_enable=True):
        def generate_compare():
            state = self.env.reset()
            step_num = 0
            total_step = int(self.dic_traffic_env_conf["EPISODE_LEN"] /
                             self.dic_traffic_env_conf["MIN_ACTION_TIME"])
            next_state = None
            while step_num < total_step:
                action_list = []
                for one_state in state:
                    action = self.meta_agent.choose_action(one_state)
                    action_list.append(action)
                next_state, reward, done, _ = self.env.step(action_list)
                state = next_state
                step_num += 1
                if done_enable and done:
                    break
            print('final inter 0: lane_vehicle_cnt ',
                  next_state[0]['lane_vehicle_cnt'])
            self.env.bulk_log()
        process_list = []
        for generate_task in self.traffic_tasks:
            work_dir = os.path.join(self.dic_path["PATH_TO_WORK"],
                                    "samples", "round_%d" % self.round_number,
                                    "generator_%s" % generate_task)
            dic_path = update_path_work(self.dic_path, work_dir)
            create_path_dir(dic_path)
            # -----------------------------------------------------
            p = Process(target=callback_func,
                        args=(self.round_number, dic_path, self.dic_exp_conf,
                              self.dic_agent_conf, self.dic_traffic_env_conf))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

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
