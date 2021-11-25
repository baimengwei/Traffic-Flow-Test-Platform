import os
import pickle

import numpy as np

from common.construct_sample import ConstructSample
from common.round_learner import RoundLearner, updater_wrapper
from collections import deque

from misc.utils import write_summary, downsample


def generator_wrapper(conf_path, round_number):
    generator = HistoryGenerator(conf_path, round_number)
    generator.generate()
    generator.generate_history()


def test_eval(conf_path, round_number):
    generator = HistoryGenerator(conf_path, round_number, is_test=True)
    generator.generate_test()


class HistoryGenerator:
    def __init__(self, conf_path, round_number, is_test=False):
        self.__conf_exp, self.__conf_agent, self.conf_traffic = \
            conf_path.load_conf_file()
        self.__conf_path = conf_path
        self.__round_number = round_number
        # create env
        env_name = self.conf_traffic.ENV_NAME
        env_package = __import__('envs.%s_env' % env_name)
        env_package = getattr(env_package, '%s_env' % env_name)
        env_class = getattr(env_package, '%sEnv' % env_name.title())
        self.__env = env_class(self.__conf_path, is_test=is_test)
        # update infos
        agents_infos = self.__env.get_agents_info()
        self.conf_traffic.set_traffic_infos(agents_infos)
        # create agents
        agent_name = self.__conf_exp.MODEL_NAME
        agent_package = __import__('algs.%s.%s_agent'
                                   % (agent_name.upper(),
                                      agent_name.lower()))
        agent_package = getattr(agent_package, '%s' % agent_name.upper())
        agent_package = getattr(agent_package, '%s_agent' % agent_name.lower())
        agent_class = getattr(agent_package, '%sAgent' % agent_name.upper())

        self.list_agent = []
        self.__list_inter = list(sorted(list(agents_infos.keys())))
        self.__conf_path.set_work_sample_total(self.__list_inter)
        for inter_name in self.__list_inter:
            # store config
            self.conf_traffic.set_intersection(inter_name)
            for i in agents_infos.keys():
                self.__conf_path.dump_conf_file(
                    self.__conf_exp, self.__conf_agent,
                    self.conf_traffic, inter_name=i)
            # create agent
            agent = agent_class(self.__conf_path, self.__round_number, inter_name)
            self.list_agent.append(agent)

        self.__list_reward = {k: 0 for k in agents_infos.keys()}
        self.__get_dim_info()

    def generate(self, done_enable=True, choice_random=True):
        state = self.__env.reset()
        step_num = 0
        total_step = int(self.conf_traffic.EPISODE_LEN /
                         self.conf_traffic.TIME_MIN_ACTION)
        next_state = None
        self.history_input = []
        while step_num < total_step:
            action_list = []
            self.__update_history_input()
            for idx, (one_state, agent) in enumerate(zip(state, self.list_agent)):
                action = agent.choose_action(one_state, self.history_input[idx],
                                             choice_random=choice_random)
                action_list.append(action)

            next_state, reward, done, _ = self.__env.step(action_list)
            self.s, self.a, self.r = state, action_list, reward

            state = next_state
            for idx, inter in enumerate(self.__list_inter):
                self.__list_reward[inter] += reward[idx]
            step_num += 1
            if done_enable and done:
                break
        print('final inter 0: lane_vehicle_cnt ',
              next_state[0]['lane_vehicle_cnt'])
        self.__env.bulk_log(self.__list_reward)

    def generate_test(self):
        for agent in self.list_agent:
            agent.load_network(self.__round_number)

        self.generate(done_enable=False, choice_random=True)
        for inter_name in self.conf_traffic.TRAFFIC_INFOS:
            write_summary(self.__conf_path, self.__round_number, inter_name)

        for inter_name in sorted(self.conf_traffic.TRAFFIC_INFOS.keys()):
            path_to_log_file = os.path.join(
                self.__conf_path.WORK_TEST, "%s.pkl" % inter_name)
            downsample(path_to_log_file)

    def generate_history(self):
        work_path = self.__conf_path.WORK_SAMPLE
        for each_file in os.listdir(work_path):
            for agent in self.list_agent:
                file_name = os.path.join(work_path, agent.inter_name + '.pkl')
                with open(file_name, "rb") as f:
                    logging_data = pickle.load(f)
                length_cnt = 0
                for each_data in logging_data:
                    if each_data["action"] != -1:
                        each_data["history"] = agent.list_history[length_cnt]
                        length_cnt += 1
                if length_cnt != len(agent.list_history):
                    raise ValueError(length_cnt, " vs ", len(agent.list_history))
                with open(file_name, "wb") as f:
                    pickle.dump(logging_data, f)

    def __get_dim_info(self):
        self.traffic_info = self.conf_traffic.TRAFFIC_INFO
        phase_dim = len(self.traffic_info['phase_links'])
        vehicle_dim = len(self.traffic_info['phase_links'])
        self.__state_dim = phase_dim + vehicle_dim
        self.__action_dim = phase_dim  # one hot represent according to phase.
        self.input_dim = self.__action_dim + 1 + self.__state_dim
        self.hidden_dim = 10
        self.__history_len = 50

    def __update_history_input(self):
        if len(self.history_input) == 0:
            width = self.__state_dim + self.__action_dim + 1
            inter_cnt = len(self.__list_inter)
            self.history_input = [[] for _ in range(inter_cnt)]
            for idx in range(inter_cnt):
                history_input = deque(maxlen=self.__history_len)
                for _ in range(self.__history_len):
                    history_input.append(np.zeros(width, ))
                self.history_input[idx] = history_input
        else:
            dic_phase_expansion = self.traffic_info['phase_lane_mapping']
            for idx, (s, a, r) in enumerate(zip(self.s, self.a, self.r)):
                # choose the action value 0-7 means phase 1-8, according to env
                s = np.append(
                    np.array(dic_phase_expansion[s["cur_phase_index"] - 1]),
                    np.array(s["lane_vehicle_cnt"]))
                a = np.array(dic_phase_expansion[a])
                msg = np.concatenate((s, a, [r]), axis=-1)
                self.history_input[idx].append(msg)


class HistoryConstructSample(ConstructSample):
    def __init__(self, conf_path, round_number):
        self.__conf_path = conf_path
        self.__round_number = round_number

        list_inters = sorted(list(self.__conf_path.load_conf_inters()))

        self.__conf_exp, _, self.conf_traffic = \
            self.__conf_path.load_conf_file(inter_name=list_inters[0])

        self.__conf_path.set_work_sample_each(
            self.__round_number, self.__conf_exp.NUM_GENERATORS, list_inters)
        self.__conf_path.set_work_sample_total(list_inters)

        self.measure_time = self.conf_traffic.TIME_MIN_ACTION
        self.__interval = self.conf_traffic.TIME_MIN_ACTION

    def construct_history(self, logging_data, time):
        state = logging_data[time]
        assert time == state["time"]
        return state["history"]

    def make_reward(self):
        gen_cnt = self.__conf_exp.NUM_GENERATORS
        for idx, each_file in enumerate(self.__conf_path.WORK_SAMPLE_EACH):
            logging_data = self.load_data(each_file)
            total_time = int(logging_data[-1]['time'] + 1)
            list_samples = []
            for time in range(0, total_time - self.measure_time + 1,
                              self.__interval):
                state = self.construct_state(
                    logging_data, self.conf_traffic.FEATURE, time)
                reward_instant, reward_average = self.construct_reward(
                    logging_data, self.conf_traffic.REWARD_INFOS, time)
                action = self.judge_action(logging_data, time)
                # THE ONLY DIFFERENCE
                history = self.construct_history(logging_data, time)

                if time + self.__interval == total_time:
                    next_state = self.construct_state(
                        logging_data, self.conf_traffic.FEATURE,
                        time + self.__interval - 1)
                else:
                    next_state = self.construct_state(
                        logging_data, self.conf_traffic.FEATURE,
                        time + self.__interval)
                sample = [state, action, next_state, reward_average,
                          reward_instant, history, time]
                list_samples.append(sample)
            self.dump_sample(
                list_samples,
                self.__conf_path.WORK_SAMPLE_TOTAL[int(idx / gen_cnt)])


class HistoryLearner(RoundLearner):
    """
    Four round phase will be call, modify phase 1.
        1. round_generate_step()
        2. round_make_samples()
        3. round_update_network()
        4. round_test_eval()
    """

    def __init__(self, conf_path, round_number):
        super().__init__(conf_path, round_number)
        pass

    def learn_round(self):
        self.round_generate_step(generator_wrapper)
        self.round_make_samples()
        self.round_update_network(updater_wrapper)
        self.round_test_eval(test_eval)

    def round_make_samples(self):
        self.conf_path.set_work_sample(self.round_number)
        cs = HistoryConstructSample(self.conf_path, self.round_number)
        cs.make_reward()
