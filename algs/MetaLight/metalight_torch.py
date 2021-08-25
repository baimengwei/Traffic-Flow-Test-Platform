import numpy as np
import warnings

from algs.FRAPPlus.frapplus_agent import FRAPPlusAgent
import copy


class MetaLightTorch:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        """
        Args:
            dic_agent_conf:
            dic_traffic_env_conf: a list please for each different task
            dic_path: a list please for each task
        """

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.policy_inter = []

        warnings.warn("assuming agent number is 2")
        self.num_agent = 2

        for i in range(self.num_agent):
            self.policy_inter.append(
                FRAPPlusAgent(
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path))
        if isinstance(self.dic_traffic_env_conf, list):
            self.group_size = self.dic_traffic_env_conf[0]["FAST_BATCH_SIZE"]
        else:
            self.group_size = self.dic_traffic_env_conf["FAST_BATCH_SIZE"]

    def choose_action(self, observations, test=False):
        """
        Args:
            observations: a list with one column
            test: False if need to output with random, else true.
        Returns:
            actions: action list, map observations
        """
        action_inter = np.zeros((len(observations)), dtype=np.int)
        for i in range(int(len(observations) / self.group_size)):
            a = i * self.group_size
            b = (i + 1) * self.group_size
            action_inter[a:b] = self.policy_inter[i].choose_action(
                observations[a:b], test)
        return action_inter

    def load_params(self, params):
        """
        each policy_inter load the same params coming from a total weights
        Args:
            params: a model's weights, dict
        Returns:
        """
        for i in range(len(self.policy_inter)):
            self.policy_inter[i].load_params(params)

    def fit(self, episodes, params, target_params):
        """
        Args:
            episodes: for different tasks(list). a buffer pool (list ..)
            params: weights for different tasks(list).
            target_params: weights for different tasks(list).
        Returns:
        """
        for i in range(len(self.policy_inter)):
            self.policy_inter[i].fit(
                episodes.episodes_inter[i],
                params=params[i],
                target_params=target_params[i])

    def update_params(self, episodes, params, lr_step, slice_index):
        """
        Args:
            episodes: for different tasks(list). a buffer pool
            params: weights for different tasks(list).
            lr_step: lr
            slice_index: samples index list
        Returns:
            new params list.
        """
        new_params = []
        for i in range(len(self.policy_inter)):
            new_params.append(
                self.policy_inter[i].update_params(
                    episodes.episodes_inter[i], params[i],
                    lr_step, slice_index))
        return new_params

    def init_params(self):
        return self.policy_inter[0].get_params()

    def get_params(self):
        params = []
        for policy in self.policy_inter:
            params.append(policy.get_params())
        return params

    def decay_epsilon(self, batch_id):
        for policy in self.policy_inter:
            policy.decay_epsilon(batch_id)

    def update_meta_params(self, episodes, slice_index, new_slice_index,
                           _params):
        """
        Args:
            episodes: for different tasks, its buffers
            slice_index: Di
            new_slice_index: Di'
            _params: meta params list, come from the local file, each item of
                list is same.
        Returns:
            new meta params * len(_params)
        """
        params = _params[0]
        tot_grads = dict(zip(params.keys(), [0] * len(params.keys())))

        for i in range(len(self.policy_inter)):
            grads = self.policy_inter[i].second_cal_grads(
                episodes.episodes_inter[i], slice_index,
                new_slice_index, params)
            for key in params.keys():
                tot_grads[key] += grads[key]

        if self.dic_agent_conf['GRADIENT_CLIP']:
            for key in tot_grads.keys():
                tot_grads[key] = np.clip(
                    tot_grads[key], -1 * self.dic_agent_conf['CLIP_SIZE'],
                    self.dic_agent_conf['CLIP_SIZE'])

        new_params = dict(zip(params.keys(),
                              [params[key]
                               - self.dic_agent_conf["BETA"] * tot_grads[key]
                               for key in params.keys()]))
        return [new_params] * len(_params)
