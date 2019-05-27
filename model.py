import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import build_dataset
from scipy import stats

class Env():
    #TODO: description here
    def _init(self, data_train, data_test, epsilon, alpha, gamma, train, learning_type, num_episodes):
        self.num_states = data_train.shape[1]
        self.max_steps_train = data_train.shape[0]
        self.max_steps_test = data_test.shape[0]
        self.Q = np.zeros((self.num_states, self.num_states))
        self.state = np.random.randint(0, self.num_states)
        self.action = np.random.randint(0, self.num_states)
        self.data_train = data_train
        self.data_test = data_test
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.train = train
        self.isEnd = False
        self.time = 0
        self.total_reward = 0
        self.timestep_reward = []
        self.learning_type = learning_type
        self.episodes_played = 0
        self.episodes_total = num_episodes
        self.reward_hist = []

    def _step(self):
        self.time += 1
        if self.train:
            if self.time >= self.max_steps_train:
                self.isEnd = True
                self.episodes_played += 1
                return
            else:
                ret = self.data_train.iloc[self.time][self.action]
                self.total_reward += ret
                self.timestep_reward.append(self.total_reward)
                # reward = stats.percentileofscore(self.reward_hist, ret, kind='mean')/100
                reward = ret
                self.reward_hist.append(ret)
                new_state = self.action
                if self.learning_type is 'Q':
                    new_action = np.argmax(self.Q[new_state, :])
                    # print(new_action)
                elif self.learning_type is 'SARSA':
                    new_action = self._greedy_espilon()
                else:
                    new_action = np.random.randint(0, self.num_states)
                if self.time == self.max_steps_train-1:
                    self.Q[self.state, self.action] += self.alpha * (reward - self.Q[self.state, new_action])
                    self.timestep_reward.append(self.total_reward)
                else:
                    self.Q[self.state, self.action] += self.alpha * (reward + (self.gamma * self.Q[new_state, new_action])
                                                                     - self.Q[self.state, self.action])
                self.state = new_state
                self.action = new_action
        else:
            if self.time >= self.max_steps_test:
                self.isEnd = True
                self.episodes_played += 1
                return
            else:
                new_action = self._greedy_espilon()
                reward = self.data_test[self.time][new_action]
                self.total_reward += reward
        return

    def _reset(self):
        self.time = 0
        self.isEnd = False
        self.total_reward = 0
        self.timestep_reward = []
        self.state = np.random.randint(0, self.num_states)
        self.action = np.random.randint(0, self.num_states)
        return

    def _greedy_espilon(self):
        self.epsilon *= 1-self.episodes_played/self.episodes_total
        if self.train or random.rand() < self.epsilon:
            action = np.argmax(self.Q[self.action, :])
        else:
            action = np.random.randint(0, self.num_states)
        return action

def _collect_total_reward(dict):
    res = []
    for k in dict.keys():
        res.append(dict[k])
    return res

if __name__ == "__main__":
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 100
    learning_type = 'SARSA'
    train = True
    output_folder = 'data/processed_data/latest_dataset'
    data_train = build_dataset.readDataFromCsv(output_folder)
    model = Env()
    model._init(data_train, data_train, epsilon, alpha, gamma, train, learning_type, episodes)
    paths = {}
    perf = []
    for episode in range(episodes):
        print('--- episode #:{} ---'.format(episode))
        while not model.isEnd:
            model._step()
        print('total reward = ', model.total_reward)
        # print(model.Q)
        paths[episode] = model.timestep_reward
        perf.append(model.total_reward)
        # plot.figure()
        plot.plot(range(model.max_steps_train), model.timestep_reward)
        model._reset()
    plot.show()
    plot.figure()
    plot.plot(range(model.episodes_total), perf)
    res = _collect_total_reward(paths)


