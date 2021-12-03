from common.trainer import Trainer
from configs.config_phaser import *


def main(args):
    """main entrance. for dqn
    """
    conf_exp, _, conf_traffic, _ = config_all(args)
    # traffic_file_list = \
    #     list(conf_traffic.TRAFFIC_CATEGORY["test_homogeneous"].keys())
    # traffic_file_list = ['cps_multi_1888']
    # traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    traffic_file_list = ['demo_train_1364']
    traffic_file_list = list(conf_traffic.TRAFFIC_CATEGORY['train_all']) + \
                        list(conf_traffic.TRAFFIC_CATEGORY['test_homogeneous'])
    print('training list:', traffic_file_list)

    # traffic_file_list_ = []
    # for traffic_file in traffic_file_list:
    #     if traffic_file.split('_')[-1] in ['2055', '1504', '1664', '1530']:
    #         traffic_file_list_.append(traffic_file)
    # traffic_file_list = traffic_file_list_

    trainer = Trainer(args, traffic_file_list)
    trainer.train()


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    print('start execute...')
    main(args)
