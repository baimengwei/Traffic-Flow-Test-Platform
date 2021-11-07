import os
import pickle
import random
import time
from misc.utils import get_file_detail


class ConfPath:

    def __init__(self, args):
        self.__model_root = "./records/weights/" + args.project
        self.__work_root = "./records/workspace/" + args.project
        self.__figure_root = "./records/figures/" + args.project
        self.__data_root = os.path.join("./data/", args.env + "_scenario/")

        _time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        _time = _time + "_No" + str(random.random()) + "_"
        _suffix = "_" + _time + args.algorithm

        self.__model = os.path.join(self.__model_root, _suffix)
        self.__work = os.path.join(self.__work_root, _suffix)
        self.__work_sample = None
        self.__work_sample_total = None
        self.__work_sample_each = None
        self.__figure = os.path.join(self.__figure_root, _suffix)
        self.__data = None

        self.__roadnet_file = None
        self.__flow_file = None
        pass

    def set_traffic_file(self, traffic_file):
        self.__data = os.path.join(self.__data_root, traffic_file)
        #
        env_name = self.__data_root.split('/')[2].split('_')[0]
        _, roadnet_file, flow_file = get_file_detail(traffic_file)
        if env_name == 'sumo':
            roadnet_file = roadnet_file + ".net.xml"
            flow_file = flow_file + ".rou.xml"
        elif env_name in ['anno', 'cityflow']:
            roadnet_file = roadnet_file + ".json"
            flow_file = flow_file + ".json"
        else:
            raise ValueError(env_name)
        self.__roadnet_file = os.path.join(self.__data_root, traffic_file, roadnet_file)
        self.__flow_file = os.path.join(self.__data_root, traffic_file, flow_file)

    def set_work(self, work_dir):
        self.__work = work_dir

    def set_work_sample(self, round_number, generate_number=None):
        if generate_number is not None:
            self.__work_sample = os.path.join(self.__work, "samples",
                                              "round_%d" % round_number,
                                              "generator_%d" % generate_number)
        elif round_number:
            self.__work_sample = os.path.join(self.__work, "samples",
                                              "round_%d" % round_number)

    def set_work_sample_total(self, list_inters):
        self.__work_sample_total = [os.path.join(self.__work, "samples",
                                                 "total_samples_%s.pkl" % inter)
                                    for inter in list_inters]

    def set_work_sample_each(self, round_number, gen_cnt, list_inters):
        self.__work_sample_each = [os.path.join(self.__work, "samples",
                                                "round_%d" % round_number,
                                                "generator_%d" % generate_number,
                                                "%s.pkl" % inter)
                                   for generate_number in range(gen_cnt)
                                   for inter in list_inters]

    def set_work_test(self, round_number):
        self.__work_test = os.path.join(self.__work, "test_round",
                                        "round_%d" % round_number)

    def set_model(self, model_dir):
        self.__model = model_dir

    def create_path_dir(self):
        os.makedirs(self.__work)
        os.makedirs(self.__model)
        os.makedirs(self.__figure)

    def dump_conf_file(self, conf_exp, conf_agent, conf_traffic):
        work_dir = self.__work
        pickle.dump(conf_exp, open(os.path.join(work_dir, "conf_exp.pkl"), mode='wb'))
        pickle.dump(conf_agent, open(os.path.join(work_dir, "conf_agent.pkl"), mode='wb'))
        pickle.dump(conf_traffic, open(os.path.join(work_dir, "conf_traffic.pkl"), mode='wb'))
        pickle.dump(self, open(os.path.join(work_dir, "conf_path.pkl"), mode='wb'))

    def load_conf_file(self):
        work_dir = self.__work
        conf_exp = pickle.load(open(os.path.join(work_dir, "conf_exp.pkl")))
        conf_agent = pickle.load(open(os.path.join(work_dir, "conf_agent.pkl")))
        conf_traffic = pickle.load(open(os.path.join(work_dir, "conf_traffic.pkl")))
        return conf_exp, conf_agent, conf_traffic

    @property
    def WORK_SAMPLE(self):
        return self.__work_sample

    @property
    def WORK_TEST(self):
        return self.__work_test

    @property
    def MODEL(self):
        return self.__model

    @property
    def WORK(self):
        return self.__work

    @property
    def FIGURE(self):
        return self.__figure

    @property
    def DATA(self):
        return self.__data

    @property
    def ROADNET_FILE(self):
        return self.__roadnet_file

    @property
    def FLOW_FILE(self):
        return self.__flow_file

    @property
    def WORK_SAMPLE_TOTAL(self):
        return self.__work_sample_total

    @property
    def WORK_SAMPLE_EACH(self):
        return self.__work_sample_each

if __name__ == '__main__':
    x = ConfPath('None')
