from common.trainer import Trainer
from configs.config_phaser import *


def main(args):
    """main entrance. for frapplus
    """
    dic_exp_conf, _, dic_traffic_env_conf, _ = config_all(args)
    traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    # traffic_file_list = ['demo_train_1364']
    conf_exp, _, conf_traffic, _ = config_all(args)
    traffic_file_list = list(conf_traffic.TRAFFIC_CATEGORY['train_all']) + \
                        list(conf_traffic.TRAFFIC_CATEGORY['test_homogeneous'])
    print(traffic_file_list)
    trainer = Trainer(args, traffic_file_list)
    trainer.train()


if __name__ == '__main__':
    """
    """
    os.chdir('../../')
    args = parse()
    print('start execute...')
    main(args)
