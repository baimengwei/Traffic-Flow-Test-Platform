from common.history_learner import HistoryLearner
from common.trainer import Trainer
from configs.config_phaser import *


def history_train(conf_path, round_number):
    print('round %s start...' % round_number)
    learner = HistoryLearner(conf_path, round_number)
    learner.learn_round()


def main(args):
    conf_exp, _, conf_traffic, _ = config_all(args)
    traffic_file_list = ['demo_train_1364']
    print('training list:', traffic_file_list)
    traffic_file_list = list(conf_traffic.TRAFFIC_CATEGORY['train_all']) + \
                        list(conf_traffic.TRAFFIC_CATEGORY['test_homogeneous']) + \
                        list(conf_traffic.TRAFFIC_CATEGORY['test_heterogeneous'])
    trainer = Trainer(args, traffic_file_list, callback=history_train)
    trainer.train()


if __name__ == '__main':
    """
    """
    os.chdir('../../')
    args = parse()
    print('start execute drqn...')
    main(args)
