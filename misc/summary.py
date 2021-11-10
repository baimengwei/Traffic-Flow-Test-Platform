import copy
import json
import pickle
from collections import defaultdict
from math import isnan
import pickle as pkl
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mlp
import pandas as pd

from misc.utils import get_total_traffic_volume

mlp.use("agg")
font = {'size': 24}
mlp.rc('font', **font)
NAN_LABEL = -1


def get_metrics(duration_list, queue_length_list, min_duration,
                min_duration_id, min_queue_length, min_queue_length_id,
                traffic_name, dict_summary, mode_name,
                save_path, num_rounds, min_duration2=None):
    validation_duration_length = 10
    minimum_round = 50 if num_rounds > 50 else 0
    duration_list = np.array(duration_list)

    # min_duration, min_duration_id = np.min(duration_list), np.argmin(
    # duration_list)
    # min_queue_length, min_queue_length_id = np.min(queue_length_list),
    # np.argmin(queue_length_list)

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
    ax.plot([0, len(duration_list)], [final_duration,
                                      final_duration], linewidth=2, color="g")
    ax.plot([conv_12, conv_12], [duration_list[conv_12],
                                 duration_list[conv_12] * 3],
            linewidth=2, color="b")
    ax.plot([conv_11, conv_11], [duration_list[conv_11],
                                 duration_list[conv_11] * 3],
            linewidth=2, color="b")
    ax.plot([0, len(duration_list)], [min_duration,
                                      min_duration], linewidth=2, color="r")
    ax.plot([min_duration_id, min_duration_id], [
        min_duration, min_duration * 3], linewidth=2, color="r")
    ax.set_title(traffic_name + "-" + str(final_duration))

    plt.legend(['duration_list', 'final_duration', 'convergence_12',
                'convergence_11', 'min_duration', 'min_duration_id'])
    plt.savefig(save_path + "/" + traffic_name + "-" + mode_name + ".png")
    plt.close()

    dict_summary["traffic_file"].append(traffic_name)
    dict_summary["traffic"].append(traffic_name.split(".xml")[0])
    dict_summary["min_queue_length"].append(min_queue_length)
    dict_summary["min_queue_length_round"].append(min_queue_length_id)
    dict_summary["min_duration"].append(min_duration)
    dict_summary["min_duration_round"].append(min_duration_id)
    dict_summary["final_duration"].append(final_duration)
    dict_summary["final_duration_std"].append(final_duration_std)
    dict_summary["convergence_1.2"].append(conv_12)
    dict_summary["convergence_1.1"].append(conv_11)
    dict_summary["nan_count"].append(nan_count)
    dict_summary["min_duration2"].append(min_duration2)

    return dict_summary


def summary_plot(traffic_performance, figure_dir, mode_name, num_rounds):
    minimum_round = 50 if num_rounds > 50 else 0
    validation_duration_length = 10
    anomaly_threshold = 1.3

    for traffic_name in traffic_performance:
        f, ax = plt.subplots(2, 1, figsize=(12, 9), dpi=100)
        performance_tmp = []
        check_list = []
        for ti in range(len(traffic_performance[traffic_name])):
            ax[0].plot(traffic_performance[traffic_name][ti][0], linewidth=2)
            validation_duration = traffic_performance[
                                      traffic_name][ti][0][
                                  -validation_duration_length:]
            final_duration = np.round(np.mean(validation_duration), decimals=2)
            if len(np.where(
                    traffic_performance[traffic_name][ti][0]
                    == NAN_LABEL)[0]) == 0:
                # and len(traffic_performance[traffic_name][ti][0]) ==
                # num_rounds:
                tmp = traffic_performance[traffic_name][ti][0]
                if len(tmp) < num_rounds:
                    tmp.extend([float("nan")] * (num_rounds -
                                                 len(traffic_performance[
                                                         traffic_name][ti][0])))
                performance_tmp.append(tmp)
                check_list.append(final_duration)
            else:
                print(
                    "the length of traffic {} is shorter than {}".format(
                        traffic_name, num_rounds))
        check_list = np.array(check_list)
        for ci in np.where(check_list > anomaly_threshold *
                           np.mean(check_list))[0][::-1]:
            del performance_tmp[ci]
            print("anomaly traffic_name:{} id:{} err:{}".format(
                traffic_name, ci, check_list[ci] - np.mean(check_list)))
        if len(performance_tmp) == 0:
            print("The result of {} is not enough for analysis.".format(
                traffic_name))
            continue
        try:
            performance_summary = np.array(performance_tmp)
            print(traffic_name, performance_summary.shape)
            ax[1].errorbar(
                x=range(len(traffic_performance[traffic_name][0][0])),
                y=np.mean(performance_summary, axis=0),
                yerr=np.std(performance_summary, axis=0))

            psm = np.mean(performance_summary, axis=0)
            validation_duration = psm[-validation_duration_length:]
            final_duration = np.round(np.mean(validation_duration), decimals=2)

            convergence = {1.2: len(psm) - 1, 1.1: len(psm) - 1}
            for j in range(minimum_round, len(psm)):
                for level in [1.2, 1.1]:
                    if max(psm[j:]) <= level * final_duration:
                        if convergence[level] > j:
                            convergence[level] = j
            ax[1].plot([0, len(psm)], [final_duration,
                                       final_duration],
                       linewidth=2, color="g")
            ax[1].plot([convergence[1.2], convergence[1.2]], [
                psm[convergence[1.2]], psm[convergence[1.2]] * 3],
                       linewidth=2, color="b")
            ax[1].plot([convergence[1.1], convergence[1.1]], [
                psm[convergence[1.1]], psm[convergence[1.1]] * 3],
                       linewidth=2, color="b")
            ax[1].set_title(traffic_name)
            ax[1].legend(['final_duration', 'convergence1.2',
                          'convergence1.1', 'performance'])
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=None, hspace=0.3)
            plt.savefig(figure_dir + "/" + traffic_name + "-" +
                        mode_name + ".png")
            plt.close()
        except BaseException:
            print("plot error")


def plot_segment_duration(round_summary, path, mode_name):
    save_path = os.path.join(path, "segments")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key in round_summary.keys():
        if "duration" in key:
            f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
            ax.plot(round_summary[key], linewidth=2, color='k')
            ax.set_title(key)
            plt.savefig(save_path + "/" + key + "-" + mode_name + ".png")
            plt.close()


def padding_duration(performance_duration):
    for traffic_name in performance_duration.keys():
        max_duration_length = max([len(x[0])
                                   for x in performance_duration[traffic_name]])
        for i, ti in enumerate(performance_duration[traffic_name]):
            performance_duration[traffic_name][i][0].extend(
                (max_duration_length - len(ti[0])) * [ti[0][-1]])

    return performance_duration


def performance_at_min_duration_round_plot(
        performance_at_min_duration_round,
        figure_dir,
        mode_name):
    for traffic_name in performance_at_min_duration_round:
        f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
        for ti in range(len(performance_at_min_duration_round[traffic_name])):
            ax.plot(
                performance_at_min_duration_round[traffic_name][ti][0],
                linewidth=2)
        plt.savefig(figure_dir + "/" + "min_duration_round" + "-" + mode_name +
                    ".png")
        plt.close()


def main(records_dir):
    dict_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }

    # summary_detail_train(project, copy.deepcopy(total_summary))

    summary_detail_test(records_dir)
    # summary_detail_test_segments(project, copy.deepcopy(total_summary))


def summary_detail_train(project, total_summary):
    # each_round_train_duration

    performance_duration = {}
    performance_at_min_duration_round = {}
    records_dir = os.path.join("records", project)
    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        print(traffic_file)

        min_queue_length = min_duration = float('inf')
        min_queue_length_id = min_duration_ind = 0

        # get run_counts to calculate the queue_length each second
        exp_conf = open(os.path.join(
            records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]
        num_rounds = dic_exp_conf["NUM_ROUNDS"]
        num_seg = run_counts // 3600

        traffic_vol = get_total_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0])
        nan_thres = 120

        duration_each_round_list = []
        queue_length_each_round_list = []

        train_round_dir = os.path.join(
            records_dir, traffic_file, "train_round")
        round_files = os.listdir(train_round_dir)
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))
        round_summary = {"round": list(range(num_rounds))}
        for round in round_files:
            try:
                round_dir = os.path.join(train_round_dir, round)

                duration_gens = 0
                queue_length_gens = 0
                cnt_gen = 0

                list_duration_seg = [float('inf')] * num_seg
                list_queue_length_seg = [float('inf')] * num_seg
                list_queue_length_id_seg = [0] * num_seg
                list_duration_id_seg = [0] * num_seg
                for gen in os.listdir(round_dir):
                    if "generator" not in gen:
                        continue

                    # summary items (queue_length) from pickle
                    gen_dir = os.path.join(
                        records_dir, traffic_file, "train_round", round, gen)
                    f = open(os.path.join(gen_dir, "inter_0.pkl"), "rb")
                    samples = pkl.load(f)

                    for sample in samples:
                        queue_length_gens += sum(sample['state']
                                                 ['lane_queue_length'])
                    sample_num = len(samples)
                    f.close()

                    # summary items (duration) from csv
                    df_vehicle_inter_0 = pd.read_csv(
                        os.path.join(round_dir, gen, "vehicle_inter_0.csv"),
                        sep=',', header=0, dtype={0: str,
                                                  1: float,
                                                  2: float},
                        names=[
                            "vehicle_id",
                            "enter_time",
                            "leave_time"])

                    duration = df_vehicle_inter_0["leave_time"].values - \
                               df_vehicle_inter_0["enter_time"].values
                    ave_duration = np.mean(
                        [time for time in duration if not isnan(time)])
                    real_traffic_vol = 0
                    nan_num = 0
                    for time in duration:
                        if not isnan(time):
                            real_traffic_vol += 1
                        else:
                            nan_num += 1
                    # print(ave_duration)
                    cnt_gen += 1
                    duration_gens += ave_duration
                    for i, interval in enumerate(range(0, run_counts, 3600)):
                        did = np.bitwise_and(
                            df_vehicle_inter_0[
                                "enter_time"].values < interval + 3600,
                            df_vehicle_inter_0["enter_time"].values > interval)
                        duration_seg = df_vehicle_inter_0["leave_time"][
                                           did].values - \
                                       df_vehicle_inter_0["enter_time"][
                                           did].values
                        ave_duration_seg = np.mean(
                            [time for time in duration_seg if not isnan(time)])
                        # print(traffic_file, round, i, ave_duration)
                        real_traffic_vol_seg = 0
                        nan_num_seg = 0
                        for time in duration_seg:
                            if not isnan(time):
                                real_traffic_vol_seg += 1
                            else:
                                nan_num_seg += 1

                        # print(real_traffic_vol, traffic_vol, traffic_vol \
                        # - real_traffic_vol, nan_num)

                        if nan_num_seg < nan_thres:
                            # if min_duration[i] > ave_duration and
                            # ave_duration > 24:
                            list_duration_seg[i] = ave_duration_seg
                            list_duration_id_seg[i] = int(round[6:])

                list_duration_seg = np.array(list_duration_seg) / cnt_gen
                for j in range(num_seg):
                    key = "min_duration-" + str(j)
                    if key not in round_summary.keys():
                        round_summary[key] = [list_duration_seg[j]]
                    else:
                        round_summary[key].append(list_duration_seg[j])

                duration_each_round_list.append(duration_gens / cnt_gen)
                queue_length_each_round_list.append(
                    queue_length_gens / cnt_gen / sample_num)

                # print(real_traffic_vol, traffic_vol, traffic_vol -
                # real_traffic_vol, nan_num)
                if min_queue_length > queue_length_gens / cnt_gen / sample_num:
                    min_queue_length = queue_length_gens / cnt_gen / sample_num
                    min_queue_length_id = int(round[6:])

                valid_flag = json.load(
                    open(os.path.join(gen_dir, "valid_flag.json")))
                if valid_flag['0']:  # temporary for one intersection
                    if min_duration > duration_gens / cnt_gen:
                        min_duration = duration_gens / cnt_gen
                        min_duration_ind = int(round[6:])
                # print(nan_num, nan_thres)

            except BaseException:
                # change anomaly label from nan to -1000 for the convenience of
                # following computation
                duration_each_round_list.append(NAN_LABEL)
                queue_length_each_round_list.append(NAN_LABEL)

        result_dir = os.path.join("summary", project, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "train_results.csv"))
        if num_seg > 1:
            round_result = pd.DataFrame(round_summary)
            round_result.to_csv(
                os.path.join(result_dir, "train_seg_results.csv"),
                index=False)
            # plot duration segment
            # 222
            plot_segment_duration(round_summary, result_dir, mode_name="train")
            duration_each_segment_list = \
                round_result.iloc[min_duration_ind][1:].values
            if ".xml" in traffic_file:
                traffic_name, traffic_time = traffic_file.split(".xml")
            elif ".json" in traffic_file:
                traffic_name, traffic_time = traffic_file.split(".json")
            if traffic_name not in performance_at_min_duration_round:
                performance_at_min_duration_round[traffic_name] = [
                    (duration_each_segment_list, traffic_time)]
            else:
                performance_at_min_duration_round[traffic_name].append(
                    (duration_each_segment_list, traffic_time))

        # total_summary
        total_summary = get_metrics(
            duration_each_round_list,
            queue_length_each_round_list,
            min_duration,
            min_duration_ind,
            min_queue_length,
            min_queue_length_id,
            traffic_file,
            total_summary,
            mode_name="train",
            save_path=result_dir,
            num_rounds=num_rounds)

        if ".xml" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".xml")
        elif ".json" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".json")
        if traffic_name not in performance_duration:
            performance_duration[traffic_name] = [
                (duration_each_round_list, traffic_time)]
        else:
            performance_duration[traffic_name].append(
                (duration_each_round_list, traffic_time))

    figure_dir = os.path.join("summary", project, "figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    if dic_exp_conf["EARLY_STOP"]:
        performance_duration = padding_duration(performance_duration)
    summary_plot(performance_duration, figure_dir, mode_name="train",
                 num_rounds=num_rounds)
    total_result = pd.DataFrame(total_summary)
    total_result.to_csv(
        os.path.join(
            "summary",
            project,
            "total_train_results.csv"))
    performance_at_min_duration_round_plot(
        performance_at_min_duration_round,
        figure_dir,
        mode_name="train")


def summary_detail_test(conf_path):
    """DISGUSTING CODE * 3
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

    dic_reward = defaultdict(lambda :[])
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


def summary_detail_baseline(project):
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

        # if "650" not in traffic_file:
        #    continue

        # get run_counts to calculate the queue_length each second
        exp_conf = open(
            os.path.join(
                records_dir,
                traffic_file,
                "exp.conf"),
            'r')
        dic_exp_conf = json.load(exp_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]

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
            "total_baseline_test_results.csv"))


def summary_reward(work_dir):
    file_names = os.listdir(work_dir)
    for file in file_names:
        if 'round' in file:
            file_path = os.path.join(work_dir, file, 'intersection_1_1.pkl')
            with open(file_path, 'rb') as f:
                content = pickle.load(f)

                print(content)


if __name__ == "__main__":
    os.chdir('../')
    dict_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }

    project = "TransferDQN"
    # summary_detail_train(project, copy.deepcopy(dict_summary))
    # summary_detail_test(project, copy.deepcopy(dict_summary))
    # summary_detail_baseline(project)
    # summary_detail_test_segments(project, copy.deepcopy(dict_summary))
    summary_reward(
        'records/workspace/FRAPPlus_Min/_08_26_10_49_19_FRAPPlus/test_round/')
