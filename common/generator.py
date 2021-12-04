from configs.config_phaser import *
from misc.utils import write_summary, downsample


class Generator:
    def __init__(self, conf_path, round_number, is_test=False):

        self.conf_exp, self.conf_agent, self.conf_traffic = \
            conf_path.load_conf_file()
        self.conf_path = conf_path
        self.round_number = round_number
        # create env
        env_name = self.conf_traffic.ENV_NAME
        env_package = __import__('envs.%s_env' % env_name)
        env_package = getattr(env_package, '%s_env' % env_name)
        env_class = getattr(env_package, '%sEnv' % env_name.title())
        self.env = env_class(self.conf_path, is_test=is_test)
        # update infos
        agents_infos = self.env.get_agents_info()
        self.conf_traffic.set_traffic_infos(agents_infos)
        # create agents
        agent_name = self.conf_exp.MODEL_NAME
        agent_package = __import__('algs.%s.%s_agent'
                                   % (agent_name.upper(),
                                      agent_name.lower()))
        agent_package = getattr(agent_package, '%s' % agent_name.upper())
        agent_package = getattr(agent_package, '%s_agent' % agent_name.lower())
        agent_class = getattr(agent_package, '%sAgent' % agent_name.upper())

        self.list_agent = []
        self.list_inter = list(sorted(list(agents_infos.keys())))
        for inter_name in self.list_inter:
            # store config
            self.conf_traffic.set_intersection(inter_name)
            self.conf_path.dump_conf_file(
                self.conf_exp, self.conf_agent,
                self.conf_traffic, inter_name=inter_name)
            # create agent
            agent = agent_class(self.conf_path, self.round_number, inter_name)
            self.list_agent.append(agent)

        self.list_reward = {k: 0 for k in agents_infos.keys()}

    def generate(self, *, done_enable=False, choice_random=True):
        state = self.env.reset()
        step_num = 0
        total_step = int(self.conf_traffic.EPISODE_LEN /
                         self.conf_traffic.TIME_MIN_ACTION)

        while step_num < total_step:
            action_list = []
            for one_state, agent in zip(state, self.list_agent):
                action = agent.choose_action(
                    one_state, choice_random=choice_random)
                action_list.append(action)
            next_state, reward, done, _ = self.env.step(action_list)
            # DEBUG
            # print(state, action_list, reward, next_state)
            state = next_state
            for idx, inter in enumerate(self.list_inter):
                self.list_reward[inter] += reward[idx]
            step_num += 1
            if step_num % 10 == 0: print('.', end='')
            if done_enable and done:
                break

        print('||final done||')
        self.env.bulk_log(reward=self.list_reward)

    def generate_test(self):
        for agent in self.list_agent:
            agent.load_network(self.round_number)

        self.generate(done_enable=False, choice_random=True)
        for inter_name in self.conf_traffic.TRAFFIC_INFOS:
            write_summary(self.conf_path, self.round_number, inter_name)

        for inter_name in sorted(self.conf_traffic.TRAFFIC_INFOS.keys()):
            path_to_log_file = os.path.join(
                self.conf_path.WORK_TEST, "%s.pkl" % inter_name)
            downsample(path_to_log_file)

    def generate_none(self):
        self.generate(done_enable=False, choice_random=False)
        for inter_name in self.conf_traffic.TRAFFIC_INFOS:
            write_summary(self.conf_path, self.round_number, inter_name)

        for inter_name in sorted(self.conf_traffic.TRAFFIC_INFOS.keys()):
            path_to_log_file = os.path.join(
                self.conf_path.WORK_TEST, "%s.pkl" % inter_name)
            downsample(path_to_log_file)
