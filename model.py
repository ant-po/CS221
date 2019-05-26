import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import build_dataset

class Env():
    #TODO: description here
    def _init(self, data_train, data_test, epsilon, alpha, gamma, train = True, learning_type = 'Q'):
        self.num_states = data_train.shape[1]
        self.max_steps_train = data_train.shape[0]
        self.max_steps_test = data_test.shape[0]
        self.Q = np.zeros((self.num_states, self.num_states))
        self.state = 1
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

    def _step(self):
        self.time += 1
        if self.train:
            if self.time >= self.max_steps_train:
                self.isEnd = True
                return
            else:
                if self.learning_type is 'Q':
                    new_state = np.argmax(self.Q[self.state, :])
                elif self.learning_type is 'SARSA':
                    new_state = self._greedy_espilon()
                else:
                    new_state = np.random.randint(0, self.num_states)
                # print(self.time, new_state)
                reward = self.data_train.iloc[self.time][new_state]
                self.total_reward += reward
                self.timestep_reward.append(self.total_reward)
                if self.time == self.max_steps_train-1:
                    self.Q[self.state, self.state] += self.alpha * (reward - self.Q[self.state, self.state])
                    self.timestep_reward.append(self.total_reward)
                else:
                    self.Q[self.state, self.state] += self.alpha * (reward + (self.gamma * self.Q[new_state, new_state])
                                                                    - self.Q[self.state, self.state])
                self.state = new_state
        else:
            if self.time >= self.max_steps_test:
                model.isEnd = True
                return
            else:
                new_state = self._greedy_espilon()
                reward = self.data_train[self.time][new_state]
                self.total_reward += reward
        return

    def _reset(self):
        self.time = 0
        self.isEnd = False
        self.total_reward = 0
        self.timestep_reward = []
        return

    def _greedy_espilon(self):
        if self.train or random.rand() < self.epsilon:
            action = np.argmax(self.Q[self.state, :])
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
    episodes = 10
    n_tests = 200
    learning_type = 'SARSA'
    train = True
    output_folder = 'data/processed_data/latest_dataset'
    data_train = build_dataset.readDataFromCsv(output_folder)
    model = Env()
    model._init(data_train, data_train, epsilon, alpha, gamma, train, learning_type)
    paths = {}
    for episode in range(episodes):
        print('--- episode #:{} ---'.format(episode))
        while not model.isEnd:
            model._step()
        print('total reward = ', model.total_reward)
        paths[episode] = model.timestep_reward
        # plot.figure()
        plot.plot(range(model.max_steps_train), model.timestep_reward)
        model._reset()
    plot.show()
    res = _collect_total_reward(paths)


