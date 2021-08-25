import multiprocessing as mp
from warnings import warn
from configs.config_phaser import update_traffic_env_conf2, update_path3
from misc.episode import BatchEpisodes, SeperateEpisode
from envs.cityflow_env import CityFlowEnv
from algs.MetaLight.metalight_agent import MetaLightAgent
import json
import os
import shutil
import random
import copy
import numpy as np
from envs.subproc_env import SubprocEnv
from misc.utils import write_summary, copy_conf_file
import pickle


class BatchSampler(object):
    def __init__(self, dic_exp_conf, dic_agent_conf, list_traffic_env_conf,
                 list_path, round_number):
        """Sample trajectories in one episode by different methods
        config list is used for task_path_map and list_traffic_env_conf,
           which will be used in reset_task function to create envs.
        Args:

        Args:
            dic_exp_conf:
            dic_agent_conf:
            list_traffic_env_conf: must be a list
            list_path: must be a list
            round_number:
        """
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.list_traffic_env_conf = list_traffic_env_conf
        self.list_path = list_path
        self.round_number = round_number

        self.num_files = len(self.list_path)
        self.envs, self.queue, self.num_files_all = self.reset_task()

        self._task_id = 0
        warn("Absolute compromise code!")
        copy_conf_file(dic_exp_conf, dic_agent_conf, list_traffic_env_conf[0],
                       list_path[0])
        # self._copy_cityflow_file()

        self.step = 0
        self.target_step = 0
        self.lr_step = 0
        self.test_step = 0

    def sample_frapplus(self, policy, batch_buffer):
        """
        Args:
            policy: a single policy for frapplus
            batch_buffer:
        Returns:
            experience for specific tasks.(episodes)
            # warn('this is for only one intersection: action')
        """
        episodes = BatchEpisodes(self.dic_agent_conf, batch_buffer)
        episodes.forget()
        # send file index for learning prepare.
        for i in range(self.num_files_all):
            self.queue.put(i)
        observations, task_ids = self.envs.reset()
        dones = [False]

        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)
            actions = actions.reshape(-1, 1)
            new_observations, rewards, dones, new_batch_ids, _ = \
                self.envs.step(actions)
            rewards /= 20
            episodes.append(observations, actions, new_observations,
                            rewards, task_ids)
            observations, task_ids = new_observations, new_batch_ids
        print('length of buffer: %d' % (len(episodes)))
        return episodes

    def sample_sotl(self, policy, tasks=None, batch_id=None):
        for i in range(len(tasks)):
            self.queue.put(i)
        for _ in range(len(tasks)):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]

        while not all(dones):
            actions = policy.choose_action(observations)
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = \
                self.envs.step(actions)
            observations, batch_ids = new_observations, new_batch_ids

        write_summary(self.list_path[0],
                      tasks, self.dic_exp_conf["EPISODE_LEN"],
                      0, self.list_traffic_env_conf[0]['FLOW_FILE'])
        self.envs.bulk_log()

    def sample_metalight(self, policy, tasks, batch_id, params=None,
                         target_params=None, episodes=None):
        """
        Args:
            policy: agent, inner has a list
            tasks: samples tasks in this round number
            batch_id: round number
            params: meta_params, come from the local saved file.
            target_params: meta_target_params, come from the local saved file.
            episodes: Should be None
        Returns:
            meta params
        the first phase for queue is tasks(task id. just a simple id).
        when the task is finished, the subprocess need get a None value for done
        """
        for i in range(len(tasks)):
            self.queue.put(i)
        for _ in range(len(tasks)):
            self.queue.put(None)

        if not episodes:
            size = int(len(tasks) /
                       self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"])
            episodes = SeperateEpisode(
                size=size,
                group_size=self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"],
                dic_agent_conf=self.dic_agent_conf)

        observations, task_ids = self.envs.reset()
        # # sort
        # observations = [observations[i] for i in task_ids]
        # task_ids = [i for i in range(len(task_ids))]

        dones = [False]

        policy.load_params(params)

        old_params = None
        old_params_update = False
        meta_update_period = 1
        meta_update = False

        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_task_ids, _ = \
                self.envs.step(actions)
            # if sort, then the trace is generated by each fixed weight.
            # else, the trace is generated by a random way.
            # sort now
            # new_observations = [new_observations[i] for i in new_task_ids]
            # new_task_ids = [i for i in range(len(new_task_ids))]

            episodes.append(observations, actions, new_observations,
                            rewards, task_ids)
            observations, task_ids = new_observations, new_task_ids

            if self.step > self.dic_agent_conf['UPDATE_START'] and \
                    self.step % self.dic_agent_conf['UPDATE_PERIOD'] == 0:
                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:
                    episodes.forget()
                if not old_params_update:
                    old_params_update = True
                    old_params = params
                # from x-> y, x'->y', yt = r + gamma * y'. save in episodes
                policy.fit(episodes, params=params,
                           target_params=target_params)
                sample_size = min(
                    self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                slice_index = random.sample(range(len(episodes)), sample_size)
                # from x, yt -> new_weights as params
                params = policy.update_params(episodes, params=copy.deepcopy(
                    params), lr_step=self.lr_step, slice_index=slice_index)
                policy.load_params(params)

                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:
                    target_params = params
                    self.target_step = 0

                # meta update
                if meta_update_period % \
                        self.dic_agent_conf["META_UPDATE_PERIOD"] == 0:
                    # from x-> y, x'->y', yt = r + gamma * y'. save in episodes
                    old_params_update = False
                    policy.fit(episodes, params=params,
                               target_params=target_params)
                    sample_size = min(
                        self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                    new_slice_index = random.sample(
                        range(len(episodes)), sample_size)
                    # from x, yt -> new_weights of **meta** as params
                    params = policy.update_meta_params(
                        episodes, slice_index,
                        new_slice_index, _params=old_params)
                    policy.load_params(params)

                meta_update_period += 1

            self.step += 1

        if not meta_update:
            policy.fit(episodes, params=params, target_params=target_params)
            sample_size = min(
                self.dic_agent_conf['SAMPLE_SIZE'],
                len(episodes))
            new_slice_index = random.sample(range(len(episodes)), sample_size)
            params = policy.update_meta_params(
                episodes, slice_index, new_slice_index, _params=old_params)
            policy.load_params(params)

            meta_update_period += 1
        policy.decay_epsilon(batch_id)
        return params[0]

        # self.envs.bulk_log()

    def sample_meta_test(self, policy, tasks, batch_id, params=None,
                         target_params=None, old_episodes=None):
        """
        in sample_meta_test, config list will have only one item.
        Args:
            policy: one policy.
            tasks: the same task in tasks due to FAST_BATCH_SIZE
            batch_id: round number
            params: meta params
            target_params: meta params
            old_episodes: experience pool
        Returns:
        """
        for i in range(len(tasks)):
            self.queue.put(i)
        for _ in range(len(tasks)):
            self.queue.put(None)

        size = int(len(tasks) /
                   self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"])
        if isinstance(policy, MetaLightAgent):
            episodes = SeperateEpisode(
                size=size,
                group_size=self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"],
                dic_agent_conf=self.dic_agent_conf)
        else:
            episodes = BatchEpisodes(
                dic_agent_conf=self.dic_agent_conf)

        # the observations size is influenced by FAST_BATCH_SIZE
        observations, batch_ids = self.envs.reset()
        dones = [False]

        if params:
            policy.load_params(params)
        while (not all(dones)) or (not self.queue.empty()):
            # collect experience.
            actions = policy.choose_action(observations)
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = \
                self.envs.step(actions)
            episodes.append(observations, actions, new_observations,
                            rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
            # run update pure dqn
            if self.step > self.dic_agent_conf['UPDATE_START'] and \
                    self.step % self.dic_agent_conf['UPDATE_PERIOD'] == 0:
                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:
                    episodes.forget()

                if isinstance(policy, MetaLightAgent):
                    policy.fit(episodes, params=[params],
                               target_params=[target_params])
                else:
                    policy.fit(episodes, params=params,
                               target_params=target_params)

                sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'],
                                  len(episodes))
                slice_index = random.sample(range(len(episodes)), sample_size)
                if isinstance(policy, MetaLightAgent):
                    params = policy.update_params(episodes,
                                                  params=copy.deepcopy(
                                                      [params]),
                                                  lr_step=self.lr_step,
                                                  slice_index=slice_index)
                    params = params[0]
                else:
                    params = policy.update_params(episodes,
                                                  params=copy.deepcopy(
                                                      params),
                                                  lr_step=self.lr_step,
                                                  slice_index=slice_index)

                policy.load_params(params)

                self.lr_step += 1
                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:
                    target_params = params
                    self.target_step = 0
            # after run pure dqn for a special task and test with test_period.
            # then, save params to local.
            if self.step > self.dic_agent_conf['UPDATE_START'] and \
                    self.step % self.dic_agent_conf['TEST_PERIOD'] == 0:
                # test with one task for all same tasks
                self.single_test_sample(policy, tasks[0],
                                        self.test_step, params=params)
                # params_dir should be the same to log_dir in the following
                # write_summary's record_dir
                params_dir = os.path.join(
                    self.list_path[0]["PATH_TO_WORK"],
                    "test_round", "tasks_round_" + str(self.test_step))
                pickle.dump(params, open(os.path.join(
                    params_dir, 'params_' + str(self.test_step) + ".pkl"),
                    'wb'))
                self.test_step += 1
            self.step += 1
        policy.decay_epsilon(batch_id)
        self.envs.bulk_log()
        return params, target_params, episodes

    def single_test_sample(self, policy, task, test_step, params):
        """
        Args:
            policy: one trained policy
            task: one task in same tasks
            test_step: test_step
            params: a dict
        Returns:
            bulk_log and test_results.csv in
            test_round/$task/test_results.csv by summary.
        """
        policy.load_params(params)
        dic_traffic_env_conf = copy.deepcopy(self.list_traffic_env_conf[0])
        dic_traffic_env_conf['TRAFFIC_FILE'] = task
        dic_path = copy.deepcopy(self.list_path[0])
        dic_path["PATH_TO_LOG"] = os.path.join(
            dic_path['PATH_TO_WORK'],
            'test_round', 'tasks_round_' + str(test_step))
        if not os.path.exists(dic_path['PATH_TO_LOG']):
            os.makedirs(dic_path['PATH_TO_LOG'])
        dic_exp_conf = copy.deepcopy(self.dic_exp_conf)

        env = CityFlowEnv(path_to_log=dic_path["PATH_TO_LOG"],
                          path_to_work_directory=dic_path["PATH_TO_DATA"],
                          dic_traffic_env_conf=dic_traffic_env_conf)
        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int(
                dic_exp_conf["EPISODE_LEN"] /
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
        write_summary(dic_path, task, self.dic_exp_conf["EPISODE_LEN"],
                      test_step, self.list_traffic_env_conf[0]['FLOW_FILE'])

    def reset_task(self):
        """regenerate new envs to avoid the engine stuck bug ?

        Returns:
            self.envs: SubprocVecEnv, self.queue for pushing index.
        """
        file_path_map = {}
        file_traffic_env_map = {}
        for path in self.list_path:
            task = path["PATH_TO_DATA"].split("/")[-1]
            file_path_map[task] = path
        for env in self.list_traffic_env_conf:
            task = env["TRAFFIC_FILE"]
            file_traffic_env_map[task] = env

        traffic_env_conf_list = []
        path_conf_list = []
        total_count = 0
        for i in range(self.num_files):
            num_batch = self.list_traffic_env_conf[i]['FAST_BATCH_SIZE']
            total_count += num_batch
            file_name = self.list_traffic_env_conf[i]['TRAFFIC_FILE']
            file_batch = [file_name] * num_batch
            for batch_index, file in enumerate(file_batch):
                dic_traffic_env_conf = copy.deepcopy(file_traffic_env_map[file])
                dic_path = copy.deepcopy(file_path_map[file])

                log_dir = os.path.join(dic_path['PATH_TO_LOG'],
                                       'fast_batch_%d' % batch_index)
                dic_path = update_path3(log_dir, dic_path)

                traffic_env_conf_list.append(dic_traffic_env_conf)
                path_conf_list.append(dic_path)

                if not os.path.exists(dic_path['PATH_TO_LOG']):
                    os.makedirs(dic_path['PATH_TO_LOG'])

        # queue: used for send task_id(file id, a simple number).
        queue = mp.Queue()
        envs = SubprocEnv(path_conf_list,
                          traffic_env_conf_list,
                          queue=queue)
        return envs, queue, total_count
