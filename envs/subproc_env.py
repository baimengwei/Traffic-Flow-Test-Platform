import os
from envs.cityflow_env import CityFlowEnv
import numpy as np
import multiprocessing as mp


class EnvWorker(mp.Process):
    def __init__(self, work_remote, dic_path, dic_traffic_env_conf, queue):
        """
        Encapsulates CityFlowEnv class, make it to multiprocess
        Args:
            work_remote: is used for get cmd and send data.
            dic_path: a single dict for this task.
            dic_traffic_env_conf: a single dict for this task.
            queue: used for send task_id(file id, a simple number).
        """
        super(EnvWorker, self).__init__()

        self.work_remote = work_remote
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.queue = queue

        self.task_id = None
        self.done = False

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_LOG"],
            path_to_work_directory=self.dic_path["PATH_TO_DATA"],
            dic_traffic_env_conf=self.dic_traffic_env_conf)

    def empty_step(self):
        observation = [
            {'cur_phase': [0], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]
        reward, done = [0.0], True
        return observation, reward, done, []

    def try_reset(self):
        """
        self.task_id is a simple number for tasks which sampled from all tasks
        Returns:
            initial state if it's the start of the task, else return false.
        """
        self.task_id = self.queue.get()
        self.done = False
        state = self.env.reset()
        return state

    def run(self):
        """
        block here, waiting for command.
        checked: reset.step.close
        """
        while True:
            command, data = self.work_remote.recv()
            if command == 'step':
                observation, reward, done, info = self.env.step(data)
                if done:
                    self.env.bulk_log()
                self.work_remote.send(
                    (observation, reward, done, self.task_id, info))
            elif command == 'reset':

                observation = self.try_reset()
                self.work_remote.send((observation, self.task_id))
            elif command == 'reset_task':
                self.env.unwrapped.reset_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.work_remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                  self.env.action_space))
            elif command == 'bulk_log':
                self.env.bulk_log()
                self.remote.send(True)
            else:
                raise NotImplementedError()


class SubprocEnv:
    def __init__(self, dic_path_list, dic_traffic_env_conf_list, queue):
        """Environment controller: single controller (agent) multiple
        environments

        Args:
            dic_path_list: a list with same or different all config list.
            dic_traffic_env_conf_list: same meaning with below
            queue: used for send file id, a simple number.
        Pipe is used for send cmd and get data.
        """
        self.dic_path_list = dic_path_list
        self.dic_traffic_env_conf_list = dic_traffic_env_conf_list
        self.queue = queue

        self.num_workers = len(self.dic_path_list)
        self._create_works()

        self.waiting = False
        self.closed = False

    def _create_works(self):
        """self.work_locals is pipe for control multi env workers in remote
        place.

        Returns:
            self.work_locals
        """
        self.work_locals, work_remotes = \
            zip(*[mp.Pipe() for _ in range(self.num_workers)])
        self.work_locals = list(self.work_locals)
        work_remotes = list(work_remotes)

        self.workers = []
        for i in range(self.num_workers):
            env = EnvWorker(work_remotes[i], self.dic_path_list[i],
                            self.dic_traffic_env_conf_list[i], self.queue)
            self.workers.append(env)

        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def step(self, actions):
        self.step_async(actions)
        msg = self.step_wait()
        return msg

    def step_async(self, actions):
        for local, action in zip(self.work_locals, actions):
            local.send(('step', action))

    def step_wait(self):
        results = [remote.recv() for remote in self.work_locals]
        observations, rewards, dones, task_ids, infos = zip(*results)
        observations = [observations[i] for i in task_ids]
        rewards = [rewards[i] for i in task_ids]
        dones = [dones[i] for i in task_ids]
        infos = [infos[i] for i in task_ids]
        task_ids = list(range(len(task_ids)))

        counter = 0
        for i, done in enumerate(dones):
            if done:
                self.work_locals[i-counter].send(('close', None))
                self.work_locals[i-counter].close()
                del self.work_locals[i-counter]
                del self.workers[i-counter]
                counter += 1

        return np.stack(observations), np.stack(
            rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for local in self.work_locals:
            local.send(('reset', None))
        results = [local.recv() for local in self.work_locals]
        observations, task_ids = zip(*results)
        observations = [observations[i] for i in task_ids]
        task_ids = list(range(len(task_ids)))
        return np.stack(observations), task_ids

    def bulk_log(self):
        for local in self.work_locals:
            local.send(('bulk_log', None))
        results = [local.recv() for local in self.work_locals]

    def reset_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True
