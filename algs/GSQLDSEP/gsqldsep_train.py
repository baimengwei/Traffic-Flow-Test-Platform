from common.once_learner import OnceLearner
from common.trainer import Trainer
from configs.config_phaser import *


def gsqldsep_train(conf_path, round_number):
    print('round %s start...' % round_number)
    learner = OnceLearner(conf_path, round_number)
    learner.learn_round()


def main(args):
    """main entrance.
    """
    # traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_17_18_2108']
    conf_exp, _, conf_traffic, _ = config_all(args)
    # traffic_file_list = list(conf_traffic.TRAFFIC_CATEGORY['train_all']) + \
    #                     list(conf_traffic.TRAFFIC_CATEGORY['test_homogeneous'])
    traffic_file_list = ['cps_multi_1888']
    trainer = Trainer(args, traffic_file_list, callback=gsqldsep_train)
    trainer.train()


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    main(args)
