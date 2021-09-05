import os
from configs.config_constant import DIC_AGENTS, DIC_ENVS
from misc.utils import write_summary, downsample, set_seed


class Generator:
    def __init__(self, round_number, dic_path, dic_exp_conf,
                 dic_agent_conf, dic_traffic_env_conf):
        self.round_number = round_number
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf

        generate_number = int(self.dic_path["PATH_TO_WORK"][-1])
        set_seed(self.dic_exp_conf["SEED"] +
                 self.round_number +
                 generate_number)

        self.agent_name = self.dic_exp_conf["MODEL_NAME"]
        self.agent = DIC_AGENTS[self.agent_name](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            round_number=self.round_number)

        self.env_name = self.dic_traffic_env_conf["ENV_NAME"]
        self.env = DIC_ENVS[self.env_name](self.dic_path,
                                           self.dic_traffic_env_conf)

    def generate(self, done_enable=True):

        state = self.env.reset()
        step_num = 0
        total_step = int(self.dic_traffic_env_conf["EPISODE_LEN"] /
                         self.dic_traffic_env_conf["MIN_ACTION_TIME"])
        next_state = None
        while step_num < total_step:
            action_list = []
            for one_state in state:
                action = self.agent.choose_action(one_state)
                action_list.append(action)
                if action is None:
                    print("a breakpoint")
            next_state, reward, done, _ = self.env.step(action_list)
            state = next_state
            step_num += 1
            if done_enable and done:
                break
        print('final inter 0: lane_num_vehicle ',
              next_state[0]['lane_num_vehicle'])
        self.env.bulk_log()

    def generate_test(self):
        self.agent.load_network('round_%d' % self.round_number)
        self.generate(done_enable=False)
        write_summary(self.dic_path, self.round_number)

        if not self.dic_exp_conf["EXP_DEBUG"]:
            for inter_name in sorted(
                    self.dic_traffic_env_conf["LANE_PHASE_INFOS"].keys()):
                path_to_log_file = os.path.join(
                    self.dic_path["PATH_TO_WORK"],
                    "%s.pkl" % inter_name
                )
                downsample(path_to_log_file)
