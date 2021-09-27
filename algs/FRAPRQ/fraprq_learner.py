from common.round_learner import *
from collections import deque


class Generator(Generator):
    def __init__(self, round_number, dic_path, dic_exp_conf,
                 dic_agent_conf, dic_traffic_env_conf):
        super().__init__(round_number, dic_path, dic_exp_conf,
                         dic_agent_conf, dic_traffic_env_conf)

        lane_phase_info = dic_traffic_env_conf["LANE_PHASE_INFO"]
        dim_feature = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]
        phase_dim = dim_feature['cur_phase'][0]
        vehicle_dim = dim_feature['lane_vehicle_cnt'][0]
        self.history_len = self.dic_agent_conf["HISTORY_LEN"]
        self.state_dim = phase_dim + vehicle_dim
        self.action_dim = phase_dim  # one hot represent according to phase.
        self.hidden_dim = self.dic_agent_conf["HIDDEN_DIM"]

    def generate(self, done_enable=True):
        self.history_input = None
        state = self.env.reset()
        step_num = 0
        total_step = int(self.dic_traffic_env_conf["EPISODE_LEN"] /
                         self.dic_traffic_env_conf["MIN_ACTION_TIME"])
        next_state = None
        while step_num < total_step:
            action_list = []
            self.update_history_input()
            for one_state in state:
                action = self.agent.choose_action(one_state, self.history_input)
                action_list.append(action)

            next_state, reward, done, _ = self.env.step(action_list)
            self.s, self.a = state, action_list

            state = next_state
            step_num += 1
            if done_enable and done:
                break
        print('final inter 0: lane_vehicle_cnt ',
              next_state[0]['lane_vehicle_cnt'])
        self.env.bulk_log()

    def update_history_input(self):
        if self.history_input is None:
            width = self.state_dim + self.action_dim
            self.history_input = deque(maxlen=self.history_len)
            for _ in range(self.history_len):
                self.history_input.append(np.zeros(width, ))
        else:
            dic_phase_expansion = self.dic_traffic_env_conf[
                "LANE_PHASE_INFO"]['phase_map']
            self.s = np.append(
                np.array(dic_phase_expansion[self.s[0]["cur_phase"][0]]),
                np.array(self.s[0]["lane_vehicle_cnt"]))
            # choose the action value 0-7 means phase 1-8, according to env
            self.a = np.array(dic_phase_expansion[self.a[0] + 1])

            msg = np.concatenate((self.s, self.a), axis=-1)
            self.history_input.append(msg)


class ConstructSample(ConstructSample):
    def __init__(self, path_to_samples, round_number, dic_traffic_env_conf):
        super().__init__(path_to_samples, round_number, dic_traffic_env_conf)

    def construct_history(self, time):
        state = self.logging_data[time]
        assert time == state["time"]
        return state["history"]

    def make_reward(self):
        self.samples = []
        for folder in os.listdir(self.path_to_samples):
            if "generator" not in folder:
                continue
            self.load_data(folder)
            list_samples = []
            total_time = int(self.logging_data[-1]['time'] + 1)
            # construct samples
            for time in range(0, total_time - self.measure_time + 1,
                              self.interval):
                state = self.construct_state(
                    self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time)
                reward_instant, reward_average = self.construct_reward(
                    self.dic_traffic_env_conf["DIC_REWARD_INFO"], time)
                action = self.judge_action(time)
                history = self.construct_history(time)

                if time + self.interval == total_time:
                    next_state = self.construct_state(
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
                        , time + self.interval - 1)
                else:
                    next_state = self.construct_state(
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                        time + self.interval)
                sample = [state, action, next_state, reward_average,
                          reward_instant, history, time]
                list_samples.append(sample)

            self.samples.extend(list_samples)

        self.dump_sample(self.samples, "")


def generator_wrapper(round_number, dic_path, dic_exp_conf,
                      dic_agent_conf, dic_traffic_env_conf):
    generator = Generator(round_number=round_number,
                          dic_path=dic_path,
                          dic_exp_conf=dic_exp_conf,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf)
    generator.generate()
    generator.agent.save_history()


def test_eval(round_number, dic_path, dic_exp_conf, dic_agent_conf,
              dic_traffic_env_conf):
    generator = Generator(round_number=round_number,
                          dic_path=dic_path,
                          dic_exp_conf=dic_exp_conf,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf)
    generator.generate_test()


class FRAPRQLearner(RoundLearner):
    """
    Four round phase will be call, cover them to update.
        1. round_generate_step()
        2. round_make_samples()
        3. round_update_network()
        4. round_test_eval()
    """

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, round_number):
        super().__init__(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                         dic_path, round_number)
        pass

    def learn_round(self):
        self.round_generate_step(generator_wrapper)
        self.round_make_samples()
        self.round_update_network(updater_wrapper)
        self.round_test_eval(test_eval)

    def round_make_samples(self):
        path_to_sample = os.path.join(
            self.dic_path["PATH_TO_WORK"],
            "samples", "round_%d" % self.round_number)

        cs = ConstructSample(
            path_to_samples=path_to_sample,
            round_number=self.round_number,
            dic_traffic_env_conf=self.dic_traffic_env_conf)
        cs.make_reward()
