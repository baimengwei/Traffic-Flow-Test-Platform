import shutil
import math
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import pickle as pkl
import os
import pandas as pd
import numpy as np
import json
import copy
from math import isnan
import matplotlib

matplotlib.use('TKAgg')


NAN_LABEL = -1


def get_metrics(duration_list, min_duration, min_duration_id,
                traffic_name, total_summary, mode_name, save_path,
                num_rounds, min_duration2=None, min_duration_log=None):
    validation_duration_length = 10
    minimum_round = 50 if num_rounds > 50 else 0
    duration_list = np.array(duration_list)

    nan_count = len(np.where(duration_list == NAN_LABEL)[0])
    validation_duration = duration_list[-validation_duration_length:]
    final_duration = np.round(
        np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(
        np.std(validation_duration[validation_duration > 0]), decimals=2)

    if nan_count == 0:
        convergence = {
            1.2: len(duration_list) - 1,
            1.1: len(duration_list) - 1}
        for j in range(minimum_round, len(duration_list)):
            for level in [1.2, 1.1]:
                if max(duration_list[j:]) <= level * final_duration:
                    if convergence[level] > j:
                        convergence[level] = j
        conv_12 = convergence[1.2]
        conv_11 = convergence[1.1]
    else:
        conv_12, conv_11 = 0, 0

    # simple plot for each training instance
    f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
    ax.plot(duration_list, linewidth=2, color='k')
    ax.plot([0, len(duration_list)], [
        final_duration, final_duration], linewidth=2, color="g")

    ax.plot([conv_12, conv_12], [duration_list[conv_12], duration_list[
        conv_12] * 3], linewidth=2, color="b")
    ax.plot([conv_11, conv_11], [duration_list[conv_11], duration_list[
        conv_11] * 3], linewidth=2, color="b")
    ax.plot([0, len(duration_list)], [min_duration, min_duration],
            linewidth=2, color="r")
    ax.plot([min_duration_id, min_duration_id], [
        min_duration, min_duration * 3], linewidth=2, color="r")

    init_dur = str(int(
        duration_list[0])) if not math.isnan(duration_list[0]) else 'NaN'
    min_dur = str(int(
        min_duration)) if not math.isnan(min_duration) else 'NaN'
    print(traffic_name, final_duration)
    final_dur = str(int(
        final_duration)) if not math.isnan(final_duration) else 'NaN'

    if min_duration == 'NaN':
        min_duration = str(int(min_duration))

    anchored_text = AnchoredText(
        "Initial_duration: %s\nMin_duration: %s\nFinal_duration: %s"
        % (init_dur, min_dur, final_dur), loc=1)
    ax.add_artist(anchored_text)

    ax.set_title(traffic_name + "-" + str(final_duration))
    plt.savefig(save_path + "/" + traffic_name + "-" + mode_name + ".png")
    figure_2 = os.path.join(os.path.dirname(save_path), 'total_figures')
    if not os.path.exists(figure_2):
        os.makedirs(figure_2)

    traffic_file = traffic_name
    if ".xml" in traffic_file:
        traffic_name, traffic_time = traffic_file.split(".xml")
    elif ".json" in traffic_file:
        traffic_name, traffic_time = traffic_file.split(".json")
    plt.savefig(figure_2 + "/" + traffic_name + ".png")
    plt.close()

    total_summary["traffic_file"].append(traffic_name)
    total_summary["traffic"].append(traffic_name.split(".xml")[0])
    total_summary["min_duration"].append(min_duration)
    total_summary["min_duration_round"].append(min_duration_id)
    total_summary["final_duration"].append(final_duration)
    total_summary["final_duration_std"].append(final_duration_std)
    total_summary["convergence_1.2"].append(conv_12)
    total_summary["convergence_1.1"].append(conv_11)
    total_summary["nan_count"].append(nan_count)
    total_summary["min_duration2"].append(min_duration2)
    if min_duration_log:
        for i in range(len(min_duration_log)):
            total_summary[
                'md_%d' % ((i + 1) * 10)].append(min_duration_log[i][0])
            total_summary[
                'md_ind_%d' % ((i + 1) * 10)].append(min_duration_log[i][1])

    return total_summary


def cal_travel_time(df_vehicle_actual_enter_leave,
                    df_vehicle_planed_enter, episode_len):
    df_vehicle_planed_enter.set_index('vehicle_id', inplace=True)
    df_vehicle_actual_enter_leave.set_index('vehicle_id', inplace=True)
    df_res = pd.concat([df_vehicle_planed_enter,
                        df_vehicle_actual_enter_leave], axis=1, sort=False)
    assert len(df_res) == len(df_vehicle_planed_enter)

    df_res["leave_time"].fillna(episode_len, inplace=True)
    df_res["travel_time"] = df_res["leave_time"] - df_res["planed_enter_time"]
    travel_time = df_res["travel_time"].mean()
    return travel_time


def summary_meta_test(project):
    ''' directly copy the write_summary'''
    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }

    path = os.path.join("records", project)
    for traffic in os.listdir(path):
        traffic_name = traffic[:traffic.find(".json") + len(".json")]
        task_name = traffic_name
        res_path = os.path.join(
            path, traffic, "test_round", task_name, "test_results.csv")
        res_summary_path = os.path.join("summary", project, "total_results")
        fig_summary_path = os.path.join("summary", project, "total_figures")
        figures_path = os.path.join("summary", project, "figures")
        if not os.path.exists(res_summary_path):
            os.makedirs(res_summary_path)
        if not os.path.exists(fig_summary_path):
            os.makedirs(fig_summary_path)
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        shutil.copy(res_path, os.path.join(
            res_summary_path, traffic[:traffic.find(".json")] + ".csv"))
        df = pd.read_csv(res_path)
        duration = df["duration"]
        min_duration = duration.min()
        min_duration_ind = duration[duration == min_duration].index[0]
        total_summary = get_metrics(
            duration, min_duration, min_duration_ind, traffic_name,
            total_summary, mode_name="test", save_path=figures_path,
            num_rounds=duration.size)
    total_result = pd.DataFrame(total_summary)

    total_result.to_csv(
        os.path.join(
            "summary",
            project,
            "total_test_results.csv"))


def summary_meta_train(path=None, batch=2):
    """
    Args:
        project:
        path: i.e. /records/meta_train/_xx_xx/learning_round:
    Returns:
        a plot and saved image
    """
    if not path or not batch:
        raise ValueError
    print('start...')
    # get
    list_round_dir_all = []
    for traffic_dir in os.listdir(path):
        round_files = os.listdir(os.path.join(path, traffic_dir))
        for round_file in round_files:
            round_dir = os.path.join(path, traffic_dir, round_file)
            list_round_dir_all.append(round_dir)
    # sort
    print('file counts: %d' % (len(list_round_dir_all)))
    list_round_dir_all.sort(key=lambda x: int(x.split('_')[-1]))
    # collect and plot
    list_duration = []

    for round_dir in list_round_dir_all:
        file_dir = os.path.join(round_dir, os.listdir(round_dir)[0])
        res_path = os.path.join(file_dir, "vehicle_inter_0.csv")
        df_vehicle_inter_0 = \
            pd.read_csv(res_path, sep=',', header=0,
                        dtype={0: str, 1: float, 2: float},
                        names=["vehicle_id", "enter_time", "leave_time"])
        duration = df_vehicle_inter_0["leave_time"].values - \
            df_vehicle_inter_0["enter_time"].values
        ave_duration = np.mean([time for time in duration if not isnan(time)])
        list_duration.append(ave_duration)

    # reformat
    list_duration = (np.array(list_duration[::2]) +
                     np.array(list_duration[1::2])) / batch
    plt.figure()
    plt.plot(list_duration)
    plt.savefig('lala.png')
    plt.show()


def summary_meta_test_custom(path=None):
    """
    Args:
        project:
        path: i.e. /records/meta_train/_xx_xx/learning_round:
    Returns:
        a plot and saved image
    """
    if not path:
        raise ValueError
    print('start...')
    # get
    list_round_dir_all = []
    for traffic_dir in os.listdir(path):
        round_files = os.path.join(path, traffic_dir)
        list_round_dir_all.append(round_files)
    # sort
    print('file counts: %d' % (len(list_round_dir_all)))
    list_round_dir_all.sort(key=lambda x: int(x.split('_')[-1]))
    # collect and plot
    list_duration = []
    for round_dir in list_round_dir_all:
        file_dir = round_dir
        res_path = os.path.join(file_dir, "vehicle_inter_0.csv")
        df_vehicle_inter_0 = \
            pd.read_csv(res_path, sep=',', header=0,
                        dtype={0: str, 1: float, 2: float},
                        names=["vehicle_id", "enter_time", "leave_time"])
        duration = df_vehicle_inter_0["leave_time"].values - \
            df_vehicle_inter_0["enter_time"].values
        ave_duration = np.mean([time for time in duration if not isnan(time)])
        list_duration.append(ave_duration)
        print('.', end='')

    # reformat
    plt.plot(list_duration)
    plt.savefig('lala.png')
    plt.show()


def summary_sotl(project):
    # each_round_train_duration
    total_summary = {
        "traffic": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": []
    }

    records_dir = os.path.join("records", project)
    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        print(traffic_file)

        # get episode_len to calculate the queue_length each second
        exp_conf = open(
            os.path.join(
                records_dir,
                traffic_file,
                "exp.conf"),
            'r')
        dic_exp_conf = json.load(exp_conf)
        episode_len = dic_exp_conf["EPISODE_LEN"]

        duration_each_round_list = []
        queue_length_each_round_list = []

        train_dir = os.path.join(records_dir, traffic_file)
        # summary items (queue_length) from pickle
        f = open(os.path.join(train_dir, "inter_0.pkl"), "rb")
        try:
            samples = pkl.load(f)
        except BaseException:
            continue
        for sample in samples:
            queue_length_each_round = sum(sample['state']['lane_queue_length'])
        f.close()

        # summary items (duration) from csv
        df_vehicle_inter_0 = pd.read_csv(
            os.path.join(
                train_dir,
                "vehicle_inter_0.csv"),
            sep=',',
            header=0,
            dtype={
                0: str,
                1: float,
                2: float},
            names=[
                "vehicle_id",
                "enter_time",
                "leave_time"])
        duration = df_vehicle_inter_0["leave_time"].values - \
            df_vehicle_inter_0["enter_time"].values
        ave_duration = np.mean([time for time in duration if not isnan(time)])
        # print(ave_duration)
        duration_each_round_list.append(ave_duration)
        ql = queue_length_each_round / len(samples)
        queue_length_each_round_list.append(ql)

        # result_dir = os.path.join(records_dir, traffic_file)
        result_dir = os.path.join("summary", project, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "test_results.csv"))
        # print(os.path.join(result_dir, "test_results.csv"))

        total_summary["traffic"].append(traffic_file)
        total_summary["min_queue_length"].append(ql)
        total_summary["min_queue_length_round"].append(0)
        total_summary["min_duration"].append(ave_duration)
        total_summary["min_duration_round"].append(0)

    total_result = pd.DataFrame(total_summary)
    total_result.to_csv(
        os.path.join(
            "summary",
            project,
            "total_sotl_test_results.csv"))


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--type', type=str, default='train')
    parse.add_argument('--batch', type=int, default='2')
    parse.add_argument('--max_round', type=int, default=5)
    args = parse.parse_args()

    if args.type == "meta_test":
        summary_meta_test(project)
    elif args.type == "sotl":
        summary_sotl(project)
    elif args.type == "meta_train":
        summary_meta_train(
            'records/meta_train/_08_05_12_05_18_FRAPPlus/meta_round',
            args.batch)
    elif args.type == 'custom':
        summary_meta_test_custom(
            'records/meta_test/hangzhou_baochu_tiyuchang_1h_12_13_1573.json_07_31_18_08_01/test_round/hangzhou_baochu_tiyuchang_1h_12_13_1573.json'
        )
    else:
        raise ValueError
