from common.round_learner import *


def generator_wrapper(round_number, dic_path, dic_exp_conf,
                      dic_agent_conf, dic_traffic_env_conf):
    generator = Generator(round_number=round_number,
                          dic_path=dic_path,
                          dic_exp_conf=dic_exp_conf,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf)
    generator.generate()
    generator.agent.save_action_prob()


class TDDDLearner(RoundLearner):
    """
        Four round phase will be call, modify phase 1.
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
