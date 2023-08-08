import numpy as np
import datetime
import csv
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(self,
                 num_of_actions: int,
                 state_shape: int | tuple[int, int] | tuple[int, int, int],
                 fit_each_steps: int,
                 test_each_steps: int,
                 num_of_test: int,
                 exploration_rate_start: float = 1,
                 exploration_rate_end: float = 5e-2,
                 exploration_rate_decay: float = 1e-4,
                 gamma: float = .75,
                 num_of_cumulative_rewards: int = 250,
                 memory_length: int = 512,
                 memory_batch_size: int = 128,
                 verbose=True,
                 *args,
                 **kwargs
                 ):
        self._num_of_actions = num_of_actions
        self._state_shape = (state_shape, ) if type(state_shape) == int else state_shape
        self._state = np.zeros(self._state_shape)
        self._fit_each_steps = fit_each_steps
        self._test_each_steps = test_each_steps
        self._num_of_test = num_of_test
        self._exploration_rate = exploration_rate_start
        self._exploration_rate_end = exploration_rate_end
        self._exploration_rate_decay = exploration_rate_decay
        self._gamma = gamma
        self._num_of_cumulative_rewards = num_of_cumulative_rewards
        self._memory_length = memory_length
        self._memory_batch_size = memory_batch_size
        self._verbose = verbose

        self._cum_rewards = []
        self._test_rewards = []

        self.cum_rewards_log = []
        self.simulation_log = []
        self.tests_log = []

        self.__tests_idx = 0
        self.__test_running = False

        self.__model_1 = self._get_model()
        self.__model_2 = self._get_model()
        p1, p2 = self.__get_predictions()
        self.__q = BaseAgent._max_min(p1, p2)

        self.__memory_buffer_x = np.zeros((self._memory_length, len(self._state)))
        self.__memory_buffer_y = np.zeros((self._memory_length, self._num_of_actions))
        self.__memory_index = 0
        self._memory_ready = False

        self.__last_history_model1 = None
        self.__last_history_model2 = None

    def __memory_add(self, x, y):
        self.__memory_buffer_x[self.__memory_index] = x
        self.__memory_buffer_y[self.__memory_index] = y
        self.__memory_index += 1
        if self.__memory_index >= self._memory_length:
            self.__memory_index = 0
            self._memory_ready = True

    def __get_predictions(self):
        s = np.expand_dims(self._state, axis=0) if len(self._state_shape) == 1 else self._state
        p1 = self.__model_1(s, training=False)[0]
        p2 = self.__model_2(s, training=False)[0]
        return p1, p2

    def _fit(self):
        x = self.__memory_buffer_x
        y = self.__memory_buffer_y
        self.last_history_model1 = self.__model_1.fit(x, y, verbose=0, batch_size=self._memory_batch_size)
        self.last_history_model2 = self.__model_2.fit(x, y, verbose=0, batch_size=self._memory_batch_size)

    def _test(self):
        self.__tests_idx += 1
        self.__test_running = True
        self.reset_state()
        self._test_rewards = []
        for _ in range(self._num_of_test):
            pt1 = self.__model_1(np.expand_dims(self._state, axis=0), training=False)[0]
            pt2 = self.__model_2(np.expand_dims(self._state, axis=0), training=False)[0]
            next_action = np.argmax(pt1 + pt2)
            self._test_rewards.append(self._get_reward(next_action))
        self.tests_log.append(np.sum(self._test_rewards))
        self.__test_running = False

    def __add_reward(self, r):
        self._cum_rewards.append(r)
        if len(self._cum_rewards) > self._num_of_cumulative_rewards:
            self._cum_rewards = self._cum_rewards[1:]
        self.cum_rewards_log.append(np.sum(self._cum_rewards))

    def __belmann(self, q1, q2, reward):
        return q1 * .1 + (reward + q2 * self._gamma) * .9

    def __add_simulation_record(self, idx, rnd, action, reward, p1, p2):
        test_string = "T" + str(self.__tests_idx)
        simulation_step = [idx]
        for i in range(0, len(self._state)):
            simulation_step.append(self._state[i])
        for i in range(0, self._num_of_actions):
            simulation_step.append(p1[i].numpy().item())
        for i in range(0, self._num_of_actions):
            simulation_step.append(p2[i].numpy().item())
        for i in range(0, self._num_of_actions):
            simulation_step.append(self.__q[i])
        simulation_step.append(rnd)
        simulation_step.append(self._exploration_rate)
        simulation_step.append(1 if rnd > self._exploration_rate else 0)
        simulation_step.append(action)
        simulation_step.append(reward)
        simulation_step.append(np.sum(self._cum_rewards))
        simulation_step.append(test_string if self.__test_running is True else "NO")
        self.simulation_log.append(simulation_step)

    def step(self, idx):
        old_state = self._state.copy()
        rnd = np.random.random()
        action = np.argmax(self.__q) if rnd > self._exploration_rate_end else np.random.randint(self._num_of_actions)
        reward = self._get_reward(action)
        self.__add_reward(reward)
        p1, p2 = self.__get_predictions()
        q2 = BaseAgent._max_min(p1, p2)
        self.__q[action] = self.__belmann(self.__q[action], np.max(q2), reward)
        self.__memory_add(old_state, self.__q)
        self.__q = q2
        if self._exploration_rate > self._exploration_rate_end:
            self._exploration_rate -= self._exploration_rate_decay
        if idx % self._fit_each_steps == 0 and self._memory_ready:
            self._fit()
        if idx % self._test_each_steps == 0 and idx > self._memory_length:
            self._test()
            if self._verbose:
                print(
                    "#", idx,
                    " CR: ", np.sum(self._cum_rewards),
                    " R: ", np.sum(self._test_rewards),
                    " L1: {:.6f}".format(self.last_history_model1.history["loss"][0]),
                    " L2: {:.6f}".format(self.last_history_model2.history["loss"][0]),
                )
        self.__add_simulation_record(idx, rnd, action, reward, p1, p2)

    def train(self, steps, show_summary=True):
        for i in range(steps):
            self.step(i)
        if show_summary:
            self.summary()

    def get_simulation_log_headers(self):
        headers = ["step"]
        for i in range(0, len(self._state)):
            headers.append("state_" + str(i))
        for i in range(0, self._num_of_actions):
            headers.append("p1_" + str(i))
        for i in range(0, self._num_of_actions):
            headers.append("p2_" + str(i))
        for i in range(0, self._num_of_actions):
            headers.append("q_" + str(i))
        headers.append("rnd")
        headers.append("er")
        headers.append("net")
        headers.append("action")
        headers.append("reward")
        headers.append("cum_reward")
        headers.append("test")
        return headers

    def get_memory_headers(self):
        headers = []
        for i in range(len(self.__memory_buffer_x)):
            headers.append("x" + str(i))
        for i in range(len(self.__memory_buffer_y)):
            headers.append("y" + str(i))
        return headers

    def save_memory_into_csv(self, append_ts=True):
        now = datetime.datetime.now()
        filename = 'memory'
        if append_ts:
            filename += "_" + now.strftime("%Y%m%d%H%M%S")
        filename += '.csv'
        headers = self.get_memory_headers()
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(headers)
            writer.writerows(self.__memory_buffer_x + self.__memory_buffer_y)

    def save_simulation_log(self, append_ts=True):
        now = datetime.datetime.now()
        filename = 'simulation_log'
        if append_ts:
            filename += "_" + now.strftime("%Y%m%d%H%M%S")
        filename += '.csv'
        headers = self.get_simulation_log_headers()
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(headers)
            writer.writerows(self.simulation_log)

    def summary(self):
        crl = self.cum_rewards_log
        cr_indexes = [i for i, _ in enumerate(crl)]
        cr_values = [x for _, x in enumerate(crl)]
        tl = self.tests_log
        tl_indexes = [i for i, _ in enumerate(tl)]
        tl_values = [x for _, x in enumerate(tl)]
        plt.figure(figsize=(7, 5))
        plt.title('Cumulative rewards')
        plt.scatter(cr_indexes, cr_values, label="DDQL Agent", s=1)
        plt.hlines(0, xmin=0, xmax=len(crl), linestyles='--', color='gray')
        plt.legend()
        plt.show()
        plt.figure(figsize=(7, 5))
        plt.title('Test Rewards')
        plt.scatter(tl_indexes, tl_values, label="DDQL Agent")
        plt.hlines(0, xmin=0, xmax=len(tl), linestyles='--', color='gray')
        plt.legend()
        plt.show()

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def _get_reward(self, action):
        pass

    @abstractmethod
    def _get_model(self):
        pass

    @staticmethod
    def _max_min(v1, v2):
        return np.where(v1 < v2, v1, v2)

    @staticmethod
    def __get_empty_episode():
        return {"rewards": [], "states": [], "won": False}
