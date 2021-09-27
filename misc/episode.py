import numpy as np
import copy
import configs.config_constant


class BatchEpisodes(object):
    def __init__(self, dic_agent_conf, old_episodes=None):
        """
        Args:
            dic_agent_conf: a dict
            old_episodes: None Please
        """
        self.dic_agent_conf = dic_agent_conf

        self.total_samples = []
        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self.tot_x = []
        self.tot_next_x = []
        if old_episodes:
            self.total_samples = self.total_samples + old_episodes.total_samples
            self.tot_x = self.tot_x + old_episodes.tot_x
            self.tot_next_x = self.tot_next_x + old_episodes.tot_next_x

        self.last_x = []
        self.last_next_x = []
        self.current_x = []
        self.current_next_x = []

    def append(self, observations, actions, new_observations,
               rewards, batch_ids):
        """
        Args:
            observations: hope for a sort list, but not
            actions: hope for a sort list, but not
            new_observations: hope for a sort list, but not
            rewards: hope for a sort list, but not
            batch_ids: only used for check whether the experiences is none
        Returns:
            store experiences and update x.
        """
        self.last_x = self.current_x
        self.last_next_x = self.current_next_x
        self.current_x = []
        self.current_next_x = []
        for observation, action, new_observation, reward, batch_id in zip(
                observations, actions, new_observations, rewards, batch_ids):
            if batch_id is None:
                continue
            self.total_samples.append(
                [observation, action, new_observation, reward, 0, 0])
            self.tot_x.append(
                observation[0]['lane_vehicle_cnt'] +
                observation[0]["cur_phase"])
            self.current_x.append(
                observation[0]['lane_vehicle_cnt'] +
                observation[0]["cur_phase"])
            self.tot_next_x.append(
                new_observation[0]['lane_vehicle_cnt'] +
                new_observation[0]["cur_phase"])
            self.current_next_x.append(
                new_observation[0]['lane_vehicle_cnt'] +
                new_observation[0]["cur_phase"])

    def get_x(self):
        return np.reshape(np.array(self.tot_x), (len(self.tot_x), -1))

    def get_next_x(self):
        return np.reshape(
            np.array(
                self.tot_next_x), (len(
                    self.tot_next_x), -1))

    def forget(self):
        """
        Returns:
            clip the samples, left the newer part.
        """
        self.total_samples = \
            self.total_samples[-1 * self.dic_agent_conf['MAX_MEMORY_LEN']:]
        self.tot_x = \
            self.tot_x[-1 * self.dic_agent_conf['MAX_MEMORY_LEN']:]
        self.tot_next_x = \
            self.tot_next_x[-1 * self.dic_agent_conf['MAX_MEMORY_LEN']:]

    def prepare_y(self, q_values):
        self.tot_y = q_values

    def get_y(self):
        return self.tot_y

    def __len__(self):
        return len(self.total_samples)


class SeperateEpisode:
    def __init__(self, size, group_size, dic_agent_conf, old_episodes=None):
        """
        Args:
            size: for samples in tasks, the different task count.
            group_size: for samples in tasks, the same task count(batch size).
            dic_agent_conf: a dict
            old_episodes: None Please
        Returns:
            episodes_inter: a list to store for each different task(size)
            for same task number, controlled by group_size
        """
        self.episodes_inter = []
        for _ in range(size):
            self.episodes_inter.append(
                BatchEpisodes(dic_agent_conf=dic_agent_conf,
                              old_episodes=old_episodes))
        self.group_size = group_size

    def append(self, observations, actions, new_observations,
               rewards, batch_ids):
        """
        Args:
            observations: all states, hope for sorted but not
            actions: all actions, hope for sorted but not
            new_observations: hope for sorted but not
            rewards: hope for sorted but not
            batch_ids: only care whether it's None.
        Returns:
            store experience in episodes_inter list.
        """
        for i in range(int(len(observations) / self.group_size)):
            a = i * self.group_size
            b = (i + 1) * self.group_size
            self.episodes_inter[i].append(observations[a: b],
                                          actions[a: b],
                                          new_observations[a: b],
                                          rewards[a: b],
                                          batch_ids)

    def forget(self):
        """
        Returns:
            clip the samples, left the newer part.
        """
        for i in range(len(self.episodes_inter)):
            self.episodes_inter[i].forget()

    def __len__(self):
        return len(self.episodes_inter[0].total_samples)
