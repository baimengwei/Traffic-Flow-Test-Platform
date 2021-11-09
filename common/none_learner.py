from multiprocessing import Process
from common.generator import Generator


def test_eval(conf_path, round_number):
    generator = Generator(conf_path, round_number, is_test=True)
    generator.generate_test()


class NoneLearner:
    def __init__(self, conf_path, round_number):
        self.conf_exp, self.conf_agent, self.conf_traffic = \
            conf_path.load_conf_file()
        self.conf_traffic.set_one_step()
        self.conf_path = conf_path
        self.round_number = round_number
        pass

    def test_round(self):
        self.__round_test_eval(test_eval)
        pass

    def __round_test_eval(self, callback_func):
        self.conf_path.set_work_test(self.round_number)
        self.conf_path.create_path_dir()

        p = Process(target=callback_func,
                    args=(self.conf_path, self.round_number,))
        p.start()
        p.join()
