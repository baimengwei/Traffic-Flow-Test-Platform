from common.trainer import Trainer
from configs.config_phaser import *


def main(args):
    """main entrance. for frapplus
    """
    dic_exp_conf, _, dic_traffic_env_conf, _ = config_all(args)
    traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    # traffic_file_list = ['demo_train_1364']

    trainer = Trainer(args, traffic_file_list)
    trainer.train()


if __name__ == '__main__':
    """
    """
    os.chdir('../../')
    args = parse()
    print('start execute...')
    main(args)
