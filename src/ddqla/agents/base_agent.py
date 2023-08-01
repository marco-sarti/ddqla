import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self,
                 num_actions: int,
                 environment: np.array,
                 fit_each_n_steps=100,
                 exploration_rate: int = 1,
                 exploration_rate_decay: float = 1e-4,
                 gamma: float = .75,
                 cumulative_rewards_max_length: int = 100,
                 memory_max_length: int = 1024,
                 memory_batch_size: int = 512,
                 allow_episode_tracking: bool = False,
                 *args,
                 **kwargs
                 ):
        self._num_actions = num_actions
        self._state = environment
        self._exploration_rate = exploration_rate
        self._exploration_rate_decay = exploration_rate_decay
        self._gamma = gamma
        self._model_1 = self._get_model(len(self._state))
        self._model_2 = self._get_model(len(self._state))
        self._fit_each_n_steps = fit_each_n_steps
        p1 = self._model_1(np.expand_dims(self._state, axis=0), training=False)[0]
        p2 = self._model_2(np.expand_dims(self._state, axis=0), training=False)[0]
        self.__q = BaseAgent._max_min(p1, p2)
        self._cum_rewards = []
        self._cum_rewards_length = cumulative_rewards_max_length
        self.cum_rewards_log = []
        self.tests_log = []
        self.environment_log = []
        self.environment_test_log = []
        self._memory_length = memory_max_length
        self._memory_buffer_x = np.zeros((self._memory_length, len(self._state)))
        self._memory_buffer_y = np.zeros((self._memory_length, self._num_actions))
        self.__memory_index = 0
        self._memory_ready = False
        self._memory_batch_size = memory_batch_size
        self.__allow_episode_tracking = allow_episode_tracking
        self.__episodes = []
        self.__actual_episode = BaseAgent.__get_empty_episode()
        self.__steps = 0

    def start_episode(self, flush=False):
        if self.__allow_episode_tracking and flush:
            self.stop_episode()
        self.__actual_episode = BaseAgent.__get_empty_episode()

    def stop_episode(self):
        self.__episodes.append(self.__actual_episode)

    def get_episodes(self):
        return self.__episodes

    def reset_episodes(self):
        self.__episodes = []
        self.__actual_episode = BaseAgent.__get_empty_episode()

    def is_memory_ready(self):
        return self._memory_ready

    def __memory_add(self, x, y):
        self._memory_buffer_x[self.__memory_index] = x
        self._memory_buffer_y[self.__memory_index] = y
        self.__memory_index += 1
        if self.__memory_index >= self._memory_length:
            self.__memory_index = 0
            self._memory_ready = True

    def __memory_get_batch(self):
        bs = self._memory_batch_size
        permuted_memory = np.random.permutation(self._memory_length)
        return self._memory_buffer_x[permuted_memory[:bs]], self._memory_buffer_y[permuted_memory[:bs]]

    def _copy_state(self, destination):
        for i in range(0, len(self._state)):
            destination[i] = self._state[i]
        return destination

    def _fit(self, batch_size):
        x = self._memory_buffer_x
        y = self._memory_buffer_y
        h1 = self._model_1.fit(x, y, verbose=0, batch_size=batch_size)
        h2 = self._model_2.fit(x, y, verbose=0, batch_size=batch_size)
        return h1, h2

    def test(self, steps):
        self.reset_state()
        rewards = 0
        self.environment_test_log.append(self._state)
        for _ in range(steps):
            pt1 = self._model_1(np.expand_dims(self._state, axis=0), training=False)[0]
            pt2 = self._model_2(np.expand_dims(self._state, axis=0), training=False)[0]
            next_action = np.argmax(pt1 + pt2)
            rewards += self._get_reward(next_action, self._state)
            self.environment_test_log.append(self._state)
        self.tests_log.append(rewards)
        return rewards

    def step(self):
        rnd = np.random.random()
        action = np.argmax(self.__q) if rnd > self._exploration_rate else np.random.randint(self._num_actions)
        reward = self._get_reward(action, self._state)
        self._cum_rewards.append(reward)
        if len(self._cum_rewards) > self._cum_rewards_length: self._cum_rewards = self._cum_rewards[1:]
        self.cum_rewards_log.append(np.sum(self._cum_rewards))
        p1 = self._model_1(np.expand_dims(self._state, axis=0), training=False)[0]
        p2 = self._model_2(np.expand_dims(self._state, axis=0), training=False)[0]
        q2 = BaseAgent._max_min(p1, p2)
        self.__q[action] = self.__q[action] * .1 + (reward + np.max(q2) * self._gamma) * .9
        self.__memory_add(self._state, self.__q)
        self.__q = q2
        self.__steps += 1
        if self.__steps % self._fit_each_n_steps == 0 and self._memory_ready:
            self._fit(self._memory_batch_size)

    def get_last_cumulative_rewards(self):
        return np.sum(self._cum_rewards)

    def summary(self):
        crl = self.cum_rewards_log
        cr_indexes = [i for i, _ in enumerate(crl)]
        cr_values = [x for _, x in enumerate(crl)]
        tl = self.tests_log
        tl_indexes = [i for i, _ in enumerate(tl)]
        tl_values = [x for _, x in enumerate(tl)]
        plt.figure(figsize=(24, 8))
        plt.title('Cumulative rewards')
        plt.scatter(cr_indexes, cr_values, label="DDQL Agent", s=1)
        plt.hlines(0, xmin=0, xmax=len(crl), linestyles='--', color='gray')
        plt.legend()
        plt.show()
        plt.figure(figsize=(24, 8))
        plt.title('Test Rewards')
        plt.scatter(tl_indexes, tl_values, label="DDQL Agent")
        plt.hlines(0, xmin=0, xmax=len(tl), linestyles='--', color='gray')
        plt.legend()
        plt.show()

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def _get_reward(self, action, environment):
        pass

    @abstractmethod
    def _get_model(self, state_features):
        pass

    @staticmethod
    def _max_min(v1, v2):
        return np.where(v1 < v2, v1, v2)

    @staticmethod
    def __get_empty_episode():
        return {"rewards": [], "states": [], "won": False}
