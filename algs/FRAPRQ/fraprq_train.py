from common.history_learner import HistoryLearner
from common.trainer import Trainer
from configs.config_phaser import *


def history_train(conf_path, round_number):
    print('round %s start...' % round_number)
    learner = HistoryLearner(conf_path, round_number)
    learner.learn_round()


def main(args):
    traffic_file_list = ['hangzhou_baochu_tiyuchang_1h_10_11_2021']
    print('training list:', traffic_file_list)
    trainer = Trainer(args, traffic_file_list, callback=history_train)
    trainer.train()


if __name__ == '__main':
    """
    """
    os.chdir('../../')
    args = parse()
    print('start execute fraprq...')
    main(args)
