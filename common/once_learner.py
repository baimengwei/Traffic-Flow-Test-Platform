from misc.utils import write_summary


class OnceGenerator:
    def __init__(self, conf_path, round_number, is_test=False):

        self.__conf_exp, self.__conf_agent, self.__conf_traffic = \
            conf_path.load_conf_file()
        self.__conf_path = conf_path
        self.__round_number = round_number
        self.__is_test = is_test
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
        self.__list_inter = list(sorted(list(agents_infos.keys())))
        for inter_name in self.__list_inter:
            # store config
            self.__conf_traffic.set_intersection(inter_name)
            self.__conf_path.dump_conf_file(
                self.__conf_exp, self.__conf_agent,
                self.__conf_traffic, inter_name=inter_name)
            # create agent
            agent = agent_class(self.__conf_path, self.__round_number, inter_name)
            self.__list_agent.append(agent)

        self.__list_reward = {k: 0 for k in agents_infos.keys()}

    def generate_train(self, *, done_enable=False, choice_random=True):
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
            if self.__is_test is False:
                self.generate_train_q(state, action_list, reward, next_state)
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
        self.generate_train(choice_random=False)
        for inter_name in self.__conf_traffic.TRAFFIC_INFOS:
            write_summary(self.__conf_path, self.__round_number, inter_name)

    def generate_save(self):
        [agent.save_metrix(self.__round_number) for agent in self.__list_agent]
        pass

    def generate_train_q(self, l_s, l_a, l_r, l_ns):
        for agent, s, a, r, ns in zip(self.__list_agent, l_s, l_a, l_r, l_ns):
            agent.train_metrix(s, a, r, ns)


class OnceLearner:
    def __init__(self, conf_path, round_number):
        self.conf_exp, self.conf_agent, self.conf_traffic = \
            conf_path.load_conf_file()
        self.conf_traffic.set_one_step()
        self.conf_path = conf_path
        self.round_number = round_number
        pass

    def learn_round(self):
        self.__round_train()
        self.__round_test()

    def __round_train(self):
        self.conf_path.set_work_sample(self.round_number, 0)
        self.conf_path.create_path_dir()
        generator = OnceGenerator(self.conf_path, self.round_number)
        generator.generate_train()
        generator.generate_save()

    def __round_test(self):
        self.conf_path.set_work_test(self.round_number)
        self.conf_path.create_path_dir()
        generator = OnceGenerator(self.conf_path, self.round_number, is_test=True)
        generator.generate_test()

