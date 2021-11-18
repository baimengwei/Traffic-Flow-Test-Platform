from common.once_learner import OnceLearner
from common.trainer import Trainer
from configs.config_phaser import *


def gsql_train(conf_path, round_number):
    print('round %s start...' % round_number)
    learner = OnceLearner(conf_path, round_number)
    learner.learn_round()


def main(args):
    """main entrance.
    """
    # traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    traffic_file_list = ['demo_train_1364']
    trainer = Trainer(args, traffic_file_list, callback=gsql_train)
    trainer.train()


if __name__ == "__main__":
    """
    """
    os.chdir('../../')
    args = parse()
    main(args)
