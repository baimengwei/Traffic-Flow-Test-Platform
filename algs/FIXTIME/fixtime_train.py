from common.tester import Tester
from configs.config_phaser import *


def main(args):
    """main entrance.
    """
    # traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    traffic_file_list = ['demo_train_1364']
    tester = Tester(args, traffic_file_list)
    tester.test()


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    main(args)
