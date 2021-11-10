from configs.config_phaser import *
from misc.utils import write_summary, downsample


class Generator:
    def __init__(self, conf_path, round_number, is_test=False):

        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file()
        self.__conf_path = conf_path
        self.__round_number = round_number
        # create env
        env_name = self.__conf_traffic.ENV_NAME
        env_package = __import__('envs.%s_env' % env_name)
        env_package = getattr(env_package, '%s_env' % env_name)
        env_class = getattr(env_package, '%sEnv' % env_name.title())
        self.__env = env_class(self.__conf_path, is_test=is_test)
        # update infos
        agents_infos = self.__env.get_agents_info()
        self.__conf_traffic.set_traffic_infos(agents_infos)
        # create agents
        agent_name = self.__conf_exp.MODEL_NAME
        agent_package = __import__('algs.%s.%s_agent'
                                   % (agent_name.upper(),
                                      agent_name.lower()))
        agent_package = getattr(agent_package, '%s' % agent_name.upper())
        agent_package = getattr(agent_package, '%s_agent' % agent_name.lower())
        agent_class = getattr(agent_package, '%sAgent' % agent_name.upper())

        self.__list_agent = []
        self.__list_inter = list(agents_infos.keys())
        for inter_name in self.__list_inter:
            # store config
            self.__conf_traffic.set_intersection(inter_name)
            for i in agents_infos.keys():
                self.__conf_path.dump_conf_file(
                    self.__conf_exp, self.__conf_agent,
                    self.__conf_traffic, inter_name=i)
            # create agent
            agent = agent_class(self.__conf_path, self.__round_number, inter_name)
            self.__list_agent.append(agent)

        self.__list_reward = {k: 0 for k in agents_infos.keys()}

    def generate(self, *, done_enable=False, choice_random=True):
        state = self.__env.reset()
        step_num = 0
        total_step = int(self.__conf_traffic.EPISODE_LEN /
                         self.__conf_traffic.TIME_MIN_ACTION)

        while step_num < total_step:
            action_list = []
            for one_state, agent in zip(state, self.__list_agent):
                action = agent.choose_action(
                    one_state, choice_random=choice_random)
                action_list.append(action)
            next_state, reward, done, _ = self.__env.step(action_list)
            # DEBUG
            # print(state, action_list, reward, next_state)
            state = next_state
            for idx, inter in enumerate(self.__list_inter):
                self.__list_reward[inter] += reward[idx]
            step_num += 1
            if step_num % 10 == 0: print('.', end='')
            if done_enable and done:
                break

        print('||final done||')
        self.__env.bulk_log(reward=self.__list_reward)

    def generate_test(self):
        for agent in self.__list_agent:
            agent.load_network(self.__round_number)

        self.generate(done_enable=False, choice_random=True)
        for inter_name in self.__conf_traffic.TRAFFIC_INFOS:
            write_summary(self.__conf_path, self.__round_number, inter_name)

        for inter_name in sorted(self.__conf_traffic.TRAFFIC_INFOS.keys()):
            path_to_log_file = os.path.join(
                self.__conf_path.WORK_TEST, "%s.pkl" % inter_name)
            downsample(path_to_log_file)

    def generate_none(self):
        self.generate(done_enable=False)
        for inter_name in self.__conf_traffic.TRAFFIC_INFOS:
            write_summary(self.__conf_path, self.__round_number, inter_name)

        for inter_name in sorted(self.__conf_traffic.TRAFFIC_INFOS.keys()):
            path_to_log_file = os.path.join(
                self.__conf_path.WORK_TEST, "%s.pkl" % inter_name)
            downsample(path_to_log_file)
