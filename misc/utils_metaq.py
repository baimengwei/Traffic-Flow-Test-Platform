import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_metadqn_train(root_dir):
    list_round_dir = os.listdir(root_dir)
    list_round_dir = sorted(list_round_dir, key=lambda x: int(x.split('_')[-1]))
    time_travel = []
    number_enter = []
    number_exit = []
    for round_dir in list_round_dir:
        round_dir_ = os.path.join(root_dir, round_dir, 'test_results.csv')
        df = pd.read_csv(round_dir_, header=None)
        time_travel.append(np.mean(df[1]))
        number_enter.append(np.mean(df[2]))
        number_exit.append(np.mean(df[3]))

    plt.plot(time_travel)
    plt.legend(['time_travel'])
    plt.savefig(os.path.join(root_dir, '../', 'time_travel.png'))
    plt.figure()
    plt.plot(number_enter)
    plt.plot(number_exit)
    plt.legend(['number_enter', 'number_exit'])
    plt.savefig(os.path.join(root_dir, '../', 'time_travel.png'))
    # plt.show()