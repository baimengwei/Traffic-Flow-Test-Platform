from configs.config_phaser import *
from misc.utils import write_summary, downsample


class Generator:
    def __init__(self, conf_path, round_number):

        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file()
        self.__conf_path = conf_path
        self.__round_number = round_number
        # create env
        env_name = self.__conf_traffic.ENV_NAME
        env_class = __import__('envs.%s_env.%sEnv' %
                               (env_name, env_name.title()))
        self.__env = env_class(self.__conf_path, self.__conf_traffic)
        # update infos
        agents_infos = self.__env.get_agents_info()
        self.__conf_traffic.set_traffic_infos(agents_infos)
        # create agents
        agent_name = self.__conf_exp.MODEL_NAME
        agent_class = __import__('algs.%s.%s_agent.%sAgent'
                                 % (agent_name.upper(),
                                    agent_name.lower(),
                                    agent_name.title()))
        self.__list_agent = []
        for inter_name in agents_infos.keys():
            self.__conf_traffic.set_intersection(inter_name)
            agent = agent_class(self.__conf_agent,
                                self.__conf_traffic,
                                self.__conf_path)
            self.__list_agent.append(agent)

    def generate(self, done_enable=True):
        state = self.__env.reset()
        step_num = 0
        total_step = int(self.__conf_traffic.EPISODE_LEN /
                         self.__conf_traffic.MIN_ACTION_TIME)
        next_state = None
        while step_num < total_step:
            action_list = []
            for one_state, agent in zip(state, self.__list_agent):
                action = agent.choose_action(one_state)
                action_list.append(action)
            next_state, reward, done, _ = self.__env.step(action_list)
            state = next_state
            step_num += 1
            if done_enable and done:
                break
        print('final inter 0: lane_vehicle_cnt ',
              next_state[0]['lane_vehicle_cnt'])
        self.__env.bulk_log()

    def generate_test(self):
        for agent in self.__list_agent:
            agent.load_network(
                agent.inter_name + '_round_%d' % self.__round_number)

        self.generate(done_enable=False)
        for inter_name in self.__conf_traffic.LANE_PHASE_INFOS:
            write_summary(self.__conf_path, self.__round_number, inter_name)

        for inter_name in sorted(self.__conf_traffic.LANE_PHASE_INFOS.keys()):
            path_to_log_file = os.path.join(
                self.__conf_path.WORK, "%s.pkl" % inter_name)
            downsample(path_to_log_file)
