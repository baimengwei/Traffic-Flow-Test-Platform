import time
from common.meta_trainer import MetaTrainer
from common.trainer import Trainer
from configs.config_phaser import *
from misc.utils import *


def main_adapt(args):
    conf_exp, _, conf_traffic, _ = config_all(args)
    traffic_file_list = list(conf_traffic.TRAFFIC_CATEGORY['train_all'].keys())
    traffic_file_list = list(sorted(traffic_file_list))
    trainer = MetaTrainer(args, traffic_file_list)
    trainer.train()
    pass


def main_test(args):
    args.algorithm = "DQN"
    conf_exp, _, conf_traffic, _ = config_all(args)
    traffic_file_list = list(conf_traffic.TRAFFIC_CATEGORY['test_homogeneous'])
    print('training list:', traffic_file_list)
    trainer = Trainer(args, traffic_file_list)
    trainer.train()


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    # args = parse()
    # main(args)
