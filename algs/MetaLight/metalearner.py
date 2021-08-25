import os
#import utils
import pickle
import os
import gc
import copy
import numpy as np
from misc.utils import write_summary
import random
from algs.SOTL.sotl_agent import SOTLAgent


class MetaLearner(object):
    def __init__(self, sampler, policy, dic_agent_conf,
                 list_traffic_env_conf, list_path):
        """
        Meta-learner incorporates MAML and MetaLight and can update
        the meta model by different learning methods.
        both sampler and policy have multi env, their inner items are list.
        Arguments:
            sampler:    sample trajectories and update model parameters
            policy:     frapplus_agent or metalight_agent
            dic_agent_conf:
            list_traffic_env_conf: must be a list
            list_path: a list
        """
        self.sampler = sampler
        self.policy = policy
        self.dic_agent_conf = dic_agent_conf
        self.list_traffic_env_conf = list_traffic_env_conf
        self.list_path = list_path

        if not isinstance(policy, SOTLAgent):
            self.meta_params = self.policy.get_params()
            self.meta_target_params = self.meta_params
        self.step_cnt = 0
        # period is used to update the target dqn
        self.period = self.dic_agent_conf['PERIOD']

    def sample_maml(self, task, batch_id):
        """
            Use MAML framework to samples trajectories before and
            after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        self.batch_id = batch_id
        tasks = [task] * self.list_traffic_env_conf[0]['FAST_BATCH_SIZE']
        # get experience from tasks and fit
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')
        learning_episodes = self.sampler.sample_maml(
            self.policy, tasks, batch_id, params=self.meta_params)
        self.policy.fit(learning_episodes, params=self.meta_params,
                        target_params=self.meta_target_params)
        # sample from experience and get new params.(0 means initial lr)
        sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'],
                          len(learning_episodes))
        slice_index = random.sample(range(len(learning_episodes)), sample_size)
        params = self.policy.update_params(
            learning_episodes, params=copy.deepcopy(self.meta_params),
            lr_step=0, slice_index=slice_index)
        # get meta experience from tasks and fit with new params.
        self.sampler.reset_task(tasks, batch_id, reset_type='meta')
        meta_episodes = self.sampler.sample_maml(
            self.policy, tasks, batch_id, params=params)
        self.policy.fit(meta_episodes, params=params,
                        target_params=self.meta_target_params)
        # sample from meta experience and get meta grads.
        sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'],
                          len(learning_episodes))
        slice_index = random.sample(range(len(learning_episodes)), sample_size)
        _grads = self.policy.cal_grads(learning_episodes, meta_episodes,
                                       slice_index=slice_index,
                                       params=self.meta_params)
        if self.dic_agent_conf['GRADIENT_CLIP']:
            for key in _grads.keys():
                _grads[key] = np.clip(
                    _grads[key], -1 * self.dic_agent_conf['CLIP_SIZE'],
                    self.dic_agent_conf['CLIP_SIZE'])
        # append meta grads to local
        with open(os.path.join(self.list_path[0]['PATH_TO_GRADIENT'],
                               "gradients_%d.pkl") % batch_id, "ab+") as f:
            pickle.dump(_grads, f, -1)
        # update meta params by params.
        self.meta_params = params
        self.step_cnt += 1
        if self.step_cnt == self.period:
            self.step_cnt = 0
            self.meta_target_params = self.meta_params
        # save meta_params to local
        # change this for testing. if saved, means use params as meta params.
        # pickle.dump(self.meta_params, open(
        #     os.path.join(self.sampler.dic_path['PATH_TO_MODEL'],
        #                  'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

    def sample_metalight(self, _tasks, batch_id):
        """
        Use MetaLight framework to samples trajectories before and after
        the update of the parameters for all the tasks.
        Then, update meta-parameters.
        Args:
            _tasks: samples tasks(different) in this round number,
                FAST_BATCH_SIZE is used for extends tasks to more.
                note that the sampler and policy is not changed.
            batch_id: round number
            meta_params and meta_target_params come from the local saved file.
        Returns:
        """
        self.batch_id = batch_id
        tasks = []
        for task in _tasks:
            tasks.extend(
                [task] * self.list_traffic_env_conf[0]['FAST_BATCH_SIZE'])
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')

        meta_params = self.sampler.sample_metalight(
            self.policy, tasks, batch_id,
            params=self.meta_params,
            target_params=self.meta_target_params)
        pickle.dump(meta_params,
                    open(os.path.join(
                        self.sampler.list_path[0]['PATH_TO_MODEL'],
                        'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

    def sample_meta_test(self, task, batch_id, old_episodes=None):
        """
        list_traffic_env_conf and list_path only have one item, thus sampler
        and policy will only have one item in their inner. but after use tasks,
        the items will be create more in sampler inner.
            Perform meta-testing with the following choice:
             1. testing within one episode
             2. offline-training in multiple episodes to obtained
                pre-trained models
            Args:
                task: a traffic file name.
                batch_id: round number. e.g. 3
                old_episodes: episodes generated and kept in former batches,
                controlled by 'MULTI_EPISODES'
        """
        self.batch_id = batch_id
        tasks = [task] * self.list_traffic_env_conf[0]['FAST_BATCH_SIZE']
        # use same task get tasks, then reset for creating envs.
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')

        self.meta_params, self.meta_target_params, episodes = \
            self.sampler.sample_meta_test(
                self.policy, tasks, batch_id, params=self.meta_params,
                target_params=self.meta_target_params,
                old_episodes=old_episodes)
        pickle.dump(self.meta_params,
                    open(os.path.join(
                        self.sampler.list_path[0]['PATH_TO_MODEL'],
                        'params_' + str(self.batch_id) + ".pkl"), 'wb'))
        return episodes
