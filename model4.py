import numpy as np
import pandas as pd
import build_dataset
import random as random

class Env():
    # TODO: description here
    def init(self, num_states, epsilon, alpha, gamma, tran_cost):
        self.num_states = num_states
        self.KB = {}
        self.current_state = np.random.randint(0, self.num_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.time = 0
        self.tran_cost = tran_cost
        self.total_reward = 0
        self.timestep_reward = []
        self.timestep_state = []
        self.reward_hist = []

    def train(self, data_train):
        # TODO: description here
        # observation : df (r1, r2, r3, time)
        # KB : dict(key: current state, val: dict(key: opt next state, val: df(observations)))
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.KB[i] = {j: pd.DataFrame()}

        for t in range(data_train.shape[0]):
            observation = data_train.iloc[t:t+1]
            for i in range(self.num_states):
                observation_after_cost = observation - self.tran_cost
                observation_after_cost.iloc[:][str(i)] += self.tran_cost
                opt_choice = int(observation_after_cost.idxmax(axis=1))
                if opt_choice in self.KB[i].keys():
                    self.KB[i][opt_choice].append(observation, ignore_index=True)
                else:
                    self.KB[i][opt_choice] = observation
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
        if not self.train or random.random() > k:
            action = np.argmax(self.Q[self.time+1, :])
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
    epsilon = 1
    tran_cost = 0
    num_states = 3
    train_set_size = 10
    train = True
    output_folder = 'data/processed_data/latest_dataset'
    data = build_dataset.readDataFromCsv(output_folder)
    data_train = data.iloc[:train_set_size][:]
    data_test = data.iloc[train_set_size+1:][:]

    # initialize the model
    model = Env()
    model.init(num_states, epsilon, alpha, gamma, tran_cost)

    # train the model
    model.train(data_train)
    print('pause')
    print(model.KB[0][0])
    # test the model
    # for t in range(data_test.count):
    #     observation = data_test.iloc[t]


    #
    #
    # for episode in range(episodes):
    #     print('')
    #     print('--- episode #:{} ---'.format(episode))
    #     while not model.isEnd:
    #         model._step()
    #     print('total reward = {0:.0%}'.format(model.total_reward))
    #     print('total time spend in states:')
    #     print('state 1: {0:.0%}'.format(model.timestep_state.count(0)/model.max_steps_train))
    #     print('state 2: {0:.0%}'.format(model.timestep_state.count(1) / model.max_steps_train))
    #     print('state 3: {0:.0%}'.format(model.timestep_state.count(2) / model.max_steps_train))
    #     # print('Final Q:', model.Q.transpose())
    #     paths[episode] = model.timestep_reward
    #     perf.append(model.total_reward)
    #     # plot.figure()
    #     # plot.plot(range(model.max_steps_train), model.timestep_reward)
    #     model._reset()
    # model.train = False
    # while not model.isEnd:
    #     model._step()
    # plot.plot(model.timestep_reward)
    # # plot.interactive(False)
    # # plot.show()
    # # plot.figure()
    # # plot.plot(range(model.episodes_total), perf)
    # # plot.show()
    # res = _collect_total_reward(paths)
    # print('pause')



