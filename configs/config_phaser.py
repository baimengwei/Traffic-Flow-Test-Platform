import argparse
import os
from configs.conf_agent import ConfAgent
from configs.conf_exp import ConfExp
from configs.conf_path import ConfPath
from configs.conf_traffic_env import ConfTrafficEnv


def config_all(args):
    """get initial four configs dict as origin.
    """
    conf_exp = ConfExp(args)
    conf_agent = ConfAgent(args)
    conf_traffic_env = ConfTrafficEnv(args)
    conf_path = ConfPath(args)

    return conf_exp, conf_agent, conf_traffic_env, conf_path


def parse():
    parser = argparse.ArgumentParser(description='RLSignal')
    # ------------------------------path.conf----------------------------------
    parser.add_argument("--project", type=str, default="project_name")
    # ------------------------------exp.conf-----------------------------------
    parser.add_argument("--algorithm", type=str, default="MetaLight")
    parser.add_argument("--train_round", type=int, default=100,
                        help="for train process")
    parser.add_argument("--task_round", type=int, default=20,
                        help="for metalight train process")
    parser.add_argument("--task_count", type=int, default=3,
                        help="for metalight train process")
    parser.add_argument("--adapt_round", type=int, default=20,
                        help="for metalight valid test")
    parser.add_argument("--num_generator", type=int, default=3)
    parser.add_argument("--num_pipeline", type=int, default=3)

    # -----------------------------traffic_env.conf---------------------------
    parser.add_argument("--episode_len", type=int, default=3600, help='for mdp')
    parser.add_argument("--done", action="store_true", help='for mdp')
    parser.add_argument("--reward_norm", type=int, default=20, help='for mdp')
    parser.add_argument("--env", type=str, default="cityflow", help='for setting')
    parser.add_argument("--time_min_action", type=int, default=10, help='action time')
    parser.add_argument("--time_yellow", type=int, default=5, help='yellow time')
    # ------------------------------------------------------------------------
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.chdir('../')
