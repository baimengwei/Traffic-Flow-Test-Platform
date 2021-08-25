import numpy as np
import torch
from collections import deque


class Runner:
    """
      This class generates batches of experiences
    """

    def __init__(self, env, model, replay_buffer=None, tasks_buffer=None,
                 burn_in=1e4, expl_noise=0.1, total_timesteps=1e6,
                 max_path_length=200, history_length=1, device='cpu'):
        """
        Args:
            env:
            model: MQL object.
            replay_buffer: Buffer object
            tasks_buffer: MultiTasksSnapshot object
            burn_in: How many time steps purely random policy is run for
            expl_noise: Std of Gaussian exploration noise
            total_timesteps: Total number of timesteps to train on
            max_path_length: e.g. 200
            history_length: e.g. 30, 15, or 20 based on args and cmd.
        """
        self.model = model
        self.env = env
        self.burn_in = burn_in
        self.device = device
        self.episode_rewards = deque(maxlen=10)
        self.episode_lens = deque(maxlen=10)
        self.replay_buffer = replay_buffer
        self.expl_noise = expl_noise
        self.total_timesteps = total_timesteps
        self.max_path_length = max_path_length
        self.hist_len = history_length
        self.tasks_buffer = tasks_buffer

    def run(self, update_iter, keep_burning=False, task_id=None,
            early_leave=200):
        """
            This function add transition to replay buffer.
            Early_leave is used in just cold start to collect more data from various tasks,
            rather than focus on just few ones
            Returns:
                # episode_timesteps means for this collecting data loop, how many steps
                # update_iter means from the outside of the function steps to append
                # episode_reward means total reward from this loop.
            save experience in self.replay_buffer and self.tasks_buffer
        """
        obs = self.env.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0
        uiter = 0
        reward_epinfos = []

        ########
        # create a queue to keep track of past rewards and actions
        ########
        rewards_hist = deque(maxlen=self.hist_len)
        actions_hist = deque(maxlen=self.hist_len)
        obsvs_hist = deque(maxlen=self.hist_len)

        next_hrews = deque(maxlen=self.hist_len)
        next_hacts = deque(maxlen=self.hist_len)
        next_hobvs = deque(maxlen=self.hist_len)

        # Given batching schema, I need to build a full seq to keep in replay buffer
        # Add to all zeros.
        zero_action = np.zeros(self.env.action_space.shape[0])
        zero_obs = np.zeros(obs.shape)
        for _ in range(self.hist_len):
            rewards_hist.append(0)
            actions_hist.append(zero_action.copy())
            obsvs_hist.append(zero_obs.copy())
            # same thing for next_h*
            next_hrews.append(0)
            next_hacts.append(zero_action.copy())
            next_hobvs.append(zero_obs.copy())
        # now add obs to the seq
        rand_action = np.random.normal(
            0, self.expl_noise, size=self.env.action_space.shape[0])
        rand_action = rand_action.clip(
            self.env.action_space.low,
            self.env.action_space.high)
        rewards_hist.append(0)
        actions_hist.append(rand_action.copy())
        obsvs_hist.append(obs.copy())

        ######
        # Start collecting data
        #####
        while not done and uiter < np.minimum(
                self.max_path_length, early_leave):
            #####
            # Convert actions_hist, rewards_hist to np.array and flatten them out
            # for example: hist =7, action_dim = 11 -->
            # np.asarray(actions_hist(7, 11)) ==> flatten ==> (77,)

            # (hist, action_dim) => (hist *action_dim,)
            np_pre_actions = np.asarray(actions_hist,
                                        dtype=np.float32).flatten()
            # (hist, )
            np_pre_rewards = np.asarray(rewards_hist,
                                        dtype=np.float32)
            # (hist, action_dim) => (hist *action_dim,)
            np_pre_obsers = np.asarray(obsvs_hist, dtype=np.float32).flatten()

            # Select action randomly or according to policy
            if keep_burning or update_iter < self.burn_in:
                action = self.env.action_space.sample()
            else:
                # select_action take into account previous action to take into account
                # previous action in selecting a new action
                action = self.model.select_action(
                    np.array(obs),
                    np.array(np_pre_actions),
                    np.array(np_pre_rewards),
                    np.array(np_pre_obsers))

                if self.expl_noise != 0:
                    action += np.random.normal(0,
                                               self.expl_noise,
                                               size=self.env.action_space.shape[0])
                    action = action.clip(
                        self.env.action_space.low,
                        self.env.action_space.high)

            # Perform action
            new_obs, reward, done, _ = self.env.step(action)
            if episode_timesteps + 1 == self.max_path_length:
                done_bool = 0
            else:
                done_bool = float(done)

            episode_reward += reward
            reward_epinfos.append(reward)

            ###############
            next_hrews.append(reward)
            next_hacts.append(action.copy())
            next_hobvs.append(obs.copy())
            # np_next_hacts and np_next_hrews are required for TD3 alg
            # (hist, action_dim) => (hist *action_dim,)
            np_next_hacts = np.asarray(next_hacts, dtype=np.float32).flatten()
            np_next_hrews = np.asarray(next_hrews, dtype=np.float32)
            np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()

            # Store data in replay buffer
            self.replay_buffer.add(
                (obs, new_obs, action, reward, done_bool,
                 np_pre_actions, np_pre_rewards, np_pre_obsers,
                 np_next_hacts, np_next_hrews, np_next_hobvs))
            # This is snapshot buffer which has short memory
            self.tasks_buffer.add(
                task_id,
                (obs, new_obs, action, reward, done_bool,
                 np_pre_actions, np_pre_rewards, np_pre_obsers,
                 np_next_hacts, np_next_hrews, np_next_hobvs))

            # new becomes old
            rewards_hist.append(reward)
            actions_hist.append(action.copy())
            obsvs_hist.append(obs.copy())

            obs = new_obs.copy()
            episode_timesteps += 1
            update_iter += 1
            uiter += 1

        info = dict()
        # episode_timesteps means for this collecting data loop, how many steps
        # update_iter means from the outside of the function steps to append
        # episode_reward means total reward from this loop.
        info['episode_timesteps'] = episode_timesteps
        info['update_iter'] = update_iter
        info['episode_reward'] = episode_reward
        info['epinfos'] = \
            [{"r": round(sum(reward_epinfos), 6),
             "l": len(reward_epinfos)}]
        return info
