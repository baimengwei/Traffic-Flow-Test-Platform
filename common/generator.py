from configs.config_phaser import *
from misc.utils import write_summary, downsample


class Generator:
    def __init__(self, round_number, dic_path, dic_exp_conf,
                 dic_agent_conf, dic_traffic_env_conf, test_flag=False):
        self.round_number = round_number
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.env_name = self.dic_traffic_env_conf["ENV_NAME"]
        self.env = DIC_ENVS[self.env_name](self.dic_path,
                                           self.dic_traffic_env_conf)

        agents_infos = self.env.get_agents_info()
        self.dic_traffic_env_conf = update_traffic_env_infos(
            self.dic_traffic_env_conf, dic_path, agents_infos)
        if not test_flag:
            work_root = os.path.join(self.dic_path['PATH_TO_WORK'],
                                     '../', '../', '../')
            copy_conf_traffic_env(self.dic_traffic_env_conf, work_root)

        self.agent_name = self.dic_exp_conf["MODEL_NAME"]

        self.list_agent = []
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

    def generate(self, done_enable=True):
        state = self.env.reset()
        step_num = 0
        total_step = int(self.dic_traffic_env_conf["EPISODE_LEN"] /
                         self.dic_traffic_env_conf["MIN_ACTION_TIME"])
        next_state = None
        while step_num < total_step:
            action_list = []
            for one_state, agent in zip(state, self.list_agent):
                action = agent.choose_action(one_state)
                action_list.append(action)
            next_state, reward, done, _ = self.env.step(action_list)
            state = next_state
            step_num += 1
            if done_enable and done:
                break
        print('final inter 0: lane_vehicle_cnt ',
              next_state[0]['lane_vehicle_cnt'])
        self.env.bulk_log()

    def generate_test(self):
        for agent in self.list_agent:
            agent.load_network(
                agent.inter_name + '_round_%d' % self.round_number)

        self.generate(done_enable=False)
        for inter_name in self.dic_traffic_env_conf["LANE_PHASE_INFOS"]:
            write_summary(self.dic_path, self.round_number, inter_name)

        if not self.dic_exp_conf["EXP_DEBUG"]:
            for inter_name in sorted(
                    self.dic_traffic_env_conf["LANE_PHASE_INFOS"].keys()):
                path_to_log_file = os.path.join(
                    self.dic_path["PATH_TO_WORK"],
                    "%s.pkl" % inter_name)
                downsample(path_to_log_file)
