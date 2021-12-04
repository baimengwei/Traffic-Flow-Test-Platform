import os
import pickle
from multiprocessing import Process
from common.construct_sample import ConstructSample
from common.generator import Generator
from common.updater import Updater


class MetaGenerator(Generator):
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
            if 'generator' in self.conf_path.WORK_SAMPLE:
                config_dir = os.path.join(self.conf_path.WORK_SAMPLE, '..')
            else:
                config_dir = self.conf_path.WORK_SAMPLE
            self.conf_path.dump_conf_file(
                self.conf_exp, self.conf_agent,
                self.conf_traffic,
                config_dir=config_dir,
                inter_name=inter_name)
            # create agent
            agent = agent_class(self.conf_path, self.round_number, inter_name)
            self.list_agent.append(agent)

        self.list_reward = {k: 0 for k in agents_infos.keys()}


class MetaConstructSample(ConstructSample):
    def __init__(self, conf_path, round_number):
        self.conf_path = conf_path
        self.round_number = round_number

        work_sample = self.conf_path.WORK_SAMPLE
        list_inters = sorted(list(self.conf_path.load_conf_inters(
            config_dir=work_sample)))

        self.conf_exp, _, self.conf_traffic = \
            self.conf_path.load_conf_file(
                config_dir=work_sample,
                inter_name=list_inters[0])

        traffic_file = self.conf_path.WORK_SAMPLE.split('/')[-1].split('traffic_')[-1]
        self.conf_path.set_work_sample_each(
            self.round_number, self.conf_exp.NUM_GENERATORS, list_inters,
            traffic_file=traffic_file)
        self.conf_path.set_work_sample_total(list_inters,
                                             round_num=round_number,
                                             traffic_file=traffic_file)

        self.measure_time = self.conf_traffic.TIME_MIN_ACTION
        self.interval = self.conf_traffic.TIME_MIN_ACTION


class MetaUpdater(Updater):
    def __init__(self, conf_path, round_number, traffic_files):
        self.conf_path = conf_path
        self.round_number = round_number
        self.traffic_files = traffic_files

        self.list_inters = self.conf_path.load_conf_inters(
            config_dir=self.conf_path.WORK_SAMPLE)
        self.conf_exp, self.conf_agent, _ = self.conf_path.load_conf_file()

        agent_name = self.conf_exp.MODEL_NAME
        agent_package = __import__('algs.%s.%s_agent'
                                   % (agent_name.upper(),
                                      agent_name.lower()))
        agent_package = getattr(agent_package, '%s' % agent_name.upper())
        agent_package = getattr(agent_package, '%s_agent' % agent_name.lower())
        agent_class = getattr(agent_package, '%sAgent' % agent_name.upper())

        self.list_agent = []
        # only support one intersection
        agent = agent_class(self.conf_path,
                            self.round_number,
                            self.list_inters[0],
                            self.traffic_files)
        self.list_agent.append(agent)

    def load_sample(self):
        self.sample_set = []
        for traffic_file in self.traffic_files:
            self.conf_path.set_work_sample_total(
                self.list_inters,
                round_num=self.round_number,
                traffic_file=traffic_file)
            sample_set = []
            for sample_file in self.conf_path.WORK_SAMPLE_TOTAL:
                sample_each = []
                f = open(sample_file, "rb")
                try:
                    while True:
                        sample_each += pickle.load(f)
                except EOFError:
                    f.close()
                    pass
                self.sample_set.append(sample_each)

    def update_network(self):
        agent = self.list_agent[0]
        for idx, sample_each in enumerate(self.sample_set):
            agent.prepare_Xs_Y(sample_each, idx)
        agent.train_network()
        agent.save_network(self.round_number)


def generator_wrapper(conf_path, round_number):
    generator = MetaGenerator(conf_path, round_number)
    generator.generate()


def updater_wrapper(conf_path, round_number, traffic_files):
    updater = MetaUpdater(
        conf_path,
        round_number,
        traffic_files
    )
    updater.load_sample()
    updater.forget_sample()
    updater.slice_sample()
    updater.update_network()
    updater.downsamples()


def test_eval(conf_path, round_number, traffic_files):
    generator = MetaGenerator(conf_path, round_number, is_test=True)
    generator.generate_test()


class MetaLearner:
    """
    """

    def __init__(self, conf_path, round_number, traffic_files):
        self.conf_exp, self.conf_agent, self.conf_traffic = \
            conf_path.load_conf_file()
        self.conf_path = conf_path
        self.round_number = round_number
        self.traffic_files = traffic_files
        pass

    def learn_round(self):
        for traffic_file in self.traffic_files:
            self.round_generate_step(generator_wrapper, traffic_file)
        for traffic_file in self.traffic_files:
            self.round_make_samples(traffic_file)
        self.round_update_network(updater_wrapper, self.traffic_files)
        for traffic_file in self.traffic_files:
            self.round_test_eval(test_eval, traffic_file)
        pass

    def round_generate_step(self, callback_func, traffic_file):
        process_list = []
        for generate_number in range(self.conf_exp.NUM_GENERATORS):
            self.conf_path.set_work_sample(self.round_number,
                                           traffic_file=traffic_file,
                                           generate_number=generate_number)
            self.conf_path.set_traffic_file(traffic_file)
            self.conf_path.create_path_dir()
            # -----------------------------------------------------
            p = Process(target=callback_func,
                        args=(self.conf_path, self.round_number,))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

    def round_make_samples(self, traffic_file):
        self.conf_path.set_work_sample(self.round_number,
                                       traffic_file=traffic_file)
        cs = MetaConstructSample(self.conf_path, self.round_number)
        cs.make_reward()

    def round_update_network(self, callback_func, traffic_files):
        p = Process(target=callback_func, args=(
            self.conf_path, self.round_number, traffic_files))
        p.start()
        p.join()

    def round_test_eval(self, callback_func, traffic_file):
        self.conf_path.set_work_test(self.round_number)
        self.conf_path.create_path_dir()

        p = Process(target=callback_func,
                    args=(self.conf_path, self.round_number, traffic_file,))
        p.start()
        p.join()
