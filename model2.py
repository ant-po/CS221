import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import build_dataset
import random as random

class Env():
    #TODO: description here
    def _init(self, data_train, data_test, epsilon, alpha, gamma, train, learning_type, num_episodes):
        self.num_states = data_train.shape[1]
        self.max_steps_train = data_train.shape[0]
        self.max_steps_test = data_test.shape[0]
        self.Q = np.zeros((self.num_states, 1))
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
        self.timestep_state = []
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
                if self.learning_type is 'Q':
                    new_state = np.argmax(self.Q)
                    # print(new_state)
                elif self.learning_type is 'SARSA':
                    new_state = self._greedy_espilon()
                else:
                    new_state = np.random.randint(0, self.num_states)

                reward = self.data_train.iloc[self.time][new_state]
                self.total_reward += reward
                self.timestep_reward.append(self.total_reward)
                self.reward_hist.append(reward)

                if self.time == self.max_steps_train-1:
                    self.Q[self.state] += self.alpha * (reward - self.Q[self.state])
                else:
                    self.Q[self.state] += self.alpha * (reward + self.gamma * self.Q[new_state]
                                                                     - self.Q[self.state])
                self.state = new_state
                self.timestep_state.append(new_state)
        else:
            if self.time >= self.max_steps_test:
                self.isEnd = True
                self.episodes_played += 1
                return
            else:
                new_state = self._greedy_espilon()
                reward = self.data_test[self.time][new_state]
                self.total_reward += reward
        return

    def _reset(self):
        self.time = 0
        self.isEnd = False
        self.total_reward = 0
        self.timestep_reward = []
        self.timestep_state = []
        self.reward_hist = []
        self.episodes_played = 0
        self.state = np.random.randint(0, self.num_states)
        return

    def _greedy_espilon(self):
        k = self.epsilon * (1-self.episodes_played/self.episodes_total)
        if not self.train or random.random() < k:
            action = np.argmax(self.Q[self.time, :])
        else:
            action = np.random.randint(0, self.num_states)
        return action

def _collect_total_reward(dict):
    res = pd.DataFrame()
    for k in dict.keys():
        temp = pd.DataFrame({k: dict[k]})
        res.append(temp)
    return res

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.5
    epsilon = 0.5
    episodes = 100
    learning_type = 'Q'
    train = True
    output_folder = 'data/processed_data/latest_dataset'
    data_train = build_dataset.readDataFromCsv(output_folder)
    model = Env()
    model._init(data_train, data_train, epsilon, alpha, gamma, train, learning_type, episodes)
    paths = {}
    perf = []
    for episode in range(episodes):
        print('')
        print('--- episode #:{} ---'.format(episode))
        while not model.isEnd:
            model._step()
        print('total reward = {0:.0%}'.format(model.total_reward))
        print('total time spend in states:')
        print('state 1: {0:.0%}'.format(model.timestep_state.count(0)/model.max_steps_train))
        print('state 2: {0:.0%}'.format(model.timestep_state.count(1) / model.max_steps_train))
        print('state 3: {0:.0%}'.format(model.timestep_state.count(2) / model.max_steps_train))
        print('Final Q:', model.Q.transpose())
        paths[episode] = model.timestep_reward
        perf.append(model.total_reward)
        # plot.figure()
        # plot.plot(range(model.max_steps_train), model.timestep_reward)
        model._reset()
    plot.interactive(False)
    # plot.show()
    # plot.figure()
    plot.plot(range(model.episodes_total), perf)
    plot.show()
    res = _collect_total_reward(paths)
    print('pause')



