import warnings

from configs.config_constant import *
from configs.config_constant_traffic import *


class ConfTrafficEnv:
    def __init__(self, args):
        self.__env = args.env
        self.__algorithm = args.algorithm
        self.__episode_len = args.episode_len
        self.__done_enable = args.done
        self.__reward_norm = args.reward_norm
        self.__time_min_action = args.time_min_action
        self.__time_yellow = args.time_yellow
        #
        self.__list_state_feature = LIST_STATE_FEATURE
        self.__reward_info = DIC_REWARD_INFO
        self.__traffic_category = TRAFFIC_CATEGORY

        self.__traffic_file = None
        self.__traffic_infos = None
        self.__inter_name = None
        self.__traffic_info = None

        if "SUMO" in args.env.upper():
            self.__port = None
        self.__preprocess()

    def __preprocess(self):
        if self.__algorithm == 'DQN':
            self.__feature = ['cur_phase', 'lane_vehicle_cnt']
        else:
            warnings.warn('using default feature, algorithm is %s' % self.__algorithm)
            self.__feature = self.__list_state_feature

    def set_traffic_file(self, traffic_file):
        self.__traffic_file = traffic_file

    def set_traffic_infos(self, infos):
        self.__traffic_infos = infos

    def set_intersection(self, inter_name):
        self.__inter_name = inter_name
        self.__traffic_info = self.__traffic_infos[inter_name]

    def set_port(self, port):
        self.__port = port

    @property
    def ENV_NAME(self):
        return self.__env

    @property
    def TIME_YELLOW(self):
        return self.__time_yellow

    @property
    def TIME_MIN_ACTION(self):
        return self.__time_min_action

    @property
    def TRAFFIC_INFOS(self):
        return self.__traffic_infos

    @property
    def TRAFFIC_INFO(self):
        return self.__traffic_info

    @property
    def FEATURE(self):
        return self.__feature

    @property
    def REWARD_INFOS(self):
        return self.__reward_info

    @property
    def TRAFFIC_CATEGORY(self):
        return self.__traffic_category
