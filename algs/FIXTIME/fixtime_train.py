from common.tester import Tester
from configs.config_phaser import *


def main(args):
    """main entrance.
    """
    # traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    # traffic_file_list = ['demo_train_1364']
    conf_exp, _, conf_traffic, _ = config_all(args)
    traffic_file_list = list(conf_traffic.TRAFFIC_CATEGORY['train_all']) + \
                        list(conf_traffic.TRAFFIC_CATEGORY['test_homogeneous'])
    traffic_file_list = ['cps_multi_1888']
    # traffic_file_list = [traffic_file_list[0]]
    for each_file in traffic_file_list:
        tester = Tester(args, [each_file])
        tester.test()


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    main(args)
