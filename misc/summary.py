import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import matplotlib as mlp
import pandas as pd

mlp.use("agg")
font = {'size': 24}
mlp.rc('font', **font)
NAN_LABEL = -1


def summary_detail_test(conf_path):
    """
    """
    conf_path.set_work_test(0)

    test_round_dir = os.path.join(conf_path.WORK_TEST, '../')
    list_files = os.listdir(test_round_dir)
    for file in list_files:
        if '.csv' in file:
            file_full = os.path.join(test_round_dir, file)
            df = pd.read_csv(file_full)
            list_duration = df['duration']
            list_vehicle_in = df['vec_in']
            list_vehicle_out = df['vec_out']

            figure = plt.figure(figsize=(16, 9))
            ax1 = figure.add_subplot(111)
            ax1.plot(list_duration)
            plt.legend(['duration'], loc='upper left')
            ax2 = ax1.twinx()
            ax2.plot(list_vehicle_in, linestyle=':')
            ax2.plot(list_vehicle_out, linestyle=':')
            plt.legend(['vehicle_in', 'vehicle_out'], loc='upper right')
            plt.show()
            figure_path = 'result_' + file.split('.')[0] + '.png'
            figure_path = os.path.join(conf_path.WORK, figure_path)
            plt.savefig(figure_path)

    dic_reward = defaultdict(lambda: [])
    list_files = list(filter(lambda x: 'round' in x, list_files))
    list_files = list(sorted(list_files, key=lambda x: int(x.split('_')[-1])))

    for file in list_files:
        if 'round' in file:
            file_full = os.path.join(test_round_dir, file, 'valid_flag.json')
            valid_info = json.load(open(file_full))
            for k in valid_info.keys():
                if 'reward' in k:
                    dic_reward[k] += [valid_info[k]]
    figure = plt.figure(figsize=(16, 9))
    ax1 = figure.add_subplot(111)
    list_inter = list(sorted(dic_reward.keys()))
    for inter in list_inter:
        ax1.plot(dic_reward[inter])
    plt.legend(list_inter)
    plt.show()
    figure_path = os.path.join(conf_path.WORK, 'reward_info.png')
    plt.savefig(figure_path)


if __name__ == "__main__":
    os.chdir('../')
