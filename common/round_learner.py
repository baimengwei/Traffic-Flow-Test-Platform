from multiprocessing import Process
from common.construct_sample import ConstructSample
from common.generator import Generator
from common.updater import Updater


def generator_wrapper(conf_path, round_number):
    generator = Generator(conf_path, round_number)
    generator.generate()


def updater_wrapper(conf_path, round_number):
    updater = Updater(
        conf_path,
        round_number
    )
    updater.load_sample()
    updater.forget_sample()
    updater.slice_sample()
    updater.update_network()
    updater.downsamples()


def test_eval(conf_path, round_number):
    generator = Generator(conf_path, round_number, is_test=True)
    generator.generate_test()


class RoundLearner:
    """
    """

    def __init__(self, conf_path, round_number):
        self.conf_exp, self.conf_agent, self.conf_traffic = \
            conf_path.load_conf_file()
        self.conf_path = conf_path
        self.round_number = round_number
        pass

    def learn_round(self):
        self.round_generate_step(generator_wrapper)
        self.round_make_samples()
        self.round_update_network(updater_wrapper)
        self.round_test_eval(test_eval)
        pass

    def round_generate_step(self, callback_func):
        process_list = []
        for generate_number in range(self.conf_exp.NUM_GENERATORS):
            self.conf_path.set_work_sample(self.round_number, generate_number)
            self.conf_path.create_path_dir()
            # -----------------------------------------------------
            p = Process(target=callback_func,
                        args=(self.conf_path, self.round_number,))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

    def round_make_samples(self):
        self.conf_path.set_work_sample(self.round_number)
        cs = ConstructSample(self.conf_path, self.round_number)
        cs.make_reward()

    def round_update_network(self, callback_func):
        p = Process(target=callback_func, args=(self.conf_path, self.round_number,))
        p.start()
        p.join()

    def round_test_eval(self, callback_func):
        self.conf_path.set_work_test(self.round_number)
        self.conf_path.create_path_dir()

        p = Process(target=callback_func,
                    args=(self.conf_path, self.round_number,))
        p.start()
        p.join()
