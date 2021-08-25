import json
import config
import copy
import os
import pickle as pkl
from cityflow_env import CityFlowEnv


class Player(object):

    def __init__(self, path, scenario):
        self.work_path = "records/" + path
        self.dic_exp_conf = \
            json.load(open(os.path.join(self.work_path, 'exp.conf')))
        self.dic_agent_conf = \
            json.load(open(os.path.join(self.work_path, 'agent.conf')))
        self.dic_traffic_env_conf = json.load(
            open(os.path.join(self.work_path, 'traffic_env.conf')))[0]
        # change key from str to int, due to json load
        str_int_key = ['phase_startLane_mapping',
                       'phase_sameStartLane_mapping',
                       'phase_roadLink_mapping']
        for _key in str_int_key:
            t = self.dic_traffic_env_conf["LANE_PHASE_INFO"][_key]
            self.dic_traffic_env_conf["LANE_PHASE_INFO"][_key] = {
                int(k): t[k] for k in t.keys()}
        # change dict to list
        for _key in self.dic_traffic_env_conf[
                "LANE_PHASE_INFO"]['phase_roadLink_mapping'].keys():
            t = self.dic_traffic_env_conf[
                "LANE_PHASE_INFO"]['phase_roadLink_mapping'][_key]
            self.dic_traffic_env_conf[
                "LANE_PHASE_INFO"]['phase_roadLink_mapping'][_key] = \
                [tuple(l) for l in t]
        self.model_path = 'model/' + path
        self.data_path = "data/scenario/" + scenario

    def play(self, round, task, if_gui=False):
        """
        load prams and get the replay file.
        Args:
            round: a number for which params to load in model dirs.
            task: the traffic file name
            if_gui: should be true to generate replay file.
        Returns:
            search records dirs, there will be a replay_round dir,
            which contains all process in this task due to env.bulk_log
        """
        dic_traffic_env_conf = copy.deepcopy(self.dic_traffic_env_conf)
        dic_traffic_env_conf['TRAFFIC_FILE'] = task
        if if_gui:
            dic_traffic_env_conf['SAVEREPLAY'] = True

        path_to_log = os.path.join(self.work_path, 'replay_round')
        if not os.path.exists(path_to_log):
            os.mkdir(path_to_log)

        env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=self.data_path,
            dic_traffic_env_conf=dic_traffic_env_conf,
            use_cityflow=True)

        policy = config.DIC_AGENTS[self.dic_exp_conf['MODEL_NAME']](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=None
        )

        params = pkl.load(open(os.path.join(self.model_path,
                                            'params_%d.pkl' % round), 'rb'))
        policy.load_params(params)
        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int(
                self.dic_exp_conf["EPISODE_LEN"] /
                dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                action = policy.choose_action([[one_state]], test=True)
                action_list.append(action[0])
            next_state, reward, done, _ = env.step(action_list)
            state = next_state
            step_num += 1
            stop_cnt += 1
        env.bulk_log()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # params path.
    rel_path = \
        "meta_train/_08_07_06_06_15FRAPPlus/hangzhou_baochu_tiyuchang_1h_17_18_2108.json_08_07_06_06_15"
    # traffic file.
    scenario = rel_path.split("/")[-1].split(".json")[0]

    round_list = [49]  # the prarms_$x.pkl, x value
    task_list = [scenario]

    player = Player(rel_path, scenario)
    for round in round_list:
        for task in task_list:
            player.play(round, task, if_gui=True)
