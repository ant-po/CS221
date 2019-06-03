import numpy as np
import pandas as pd
import build_dataset
import datetime as time

class Env():
    # TODO: description here
    def __init__(self, num_states, epsilon, alpha, gamma, tran_cost):
        self.num_states = num_states
        self.curr_state = 0
        self.next_state = np.random.randint(0, self.num_states)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.time = 0
        self.tran_cost = tran_cost
        self.total_reward = 0
        self.error = []
        self.timestep_state = []
        self.reward_hist = []
        self.KB = {}
        self.centroids = {}
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.KB[i] = {j: pd.DataFrame()}
                self.centroids[i] = {j: pd.Series()}
        self.timestep_state.append(self.curr_state)
        self.reward_hist.append(0)
        self.error.append(0)
        return

    def train(self, data):
        # TODO: description here
        # observation : df (r1, r2, r3, time)
        # KB : dict(key: current state, val: dict(key: opt next state, val: df(observations)))
        prior_observation = data.iloc[0:1]
        for t in range(1, data.shape[0]):
            new_observation = data.iloc[t:t + 1]
            for i in range(self.num_states):
                observation_after_cost = new_observation.iloc[:, :-1] - self.tran_cost
                observation_after_cost.iloc[:][str(i)] += self.tran_cost
                opt_choice = int(observation_after_cost.idxmax(axis=1))
                if opt_choice in self.KB[i].keys():
                    self.KB[i][opt_choice] = self.KB[i][opt_choice].append(prior_observation, ignore_index=True)
                else:
                    self.KB[i][opt_choice] = prior_observation
            prior_observation = new_observation
        self.updateCentroids()
        return

    def updateCentroids(self):
        # TODO: description here
        for i in range(self.num_states):
            for j in range(self.num_states):
                data_slice = self.KB[i][j]
                x = data_slice.iloc[:, :-1].mul(data_slice.loc[:, 'time'], axis=0).sum(
                    axis=0) / data_slice.iloc[:, 3].sum()
                self.centroids[i][j] = pd.DataFrame(x).transpose()

    def predict(self, observation):
        # TODO: description here
        self.curr_state = self.next_state
        # find the closest centroid to the observation
        distances = self.distToCentroid(observation)
        self.next_state = np.argmin(distances)
        return

    def distToCentroid(self, observation):
        # TODO: description here
        distances = []
        for i in range(self.num_states):
            xyz = ['0', '1', '2']
            dist = np.sum((observation[xyz].values - self.centroids[self.curr_state][i][xyz].values) ** 2, axis=1)
            distances.append(dist)
        return distances

    def output(self, data, filename):
        # output final state of the model to Excel file
        output = data
        output.insert(output.shape[1], column='Model performance', value=self.reward_hist)
        output.insert(output.shape[1], column='Model states', value=self.timestep_state)
        output.insert(output.shape[1], column='Model error', value=self.error)
        output.to_excel(filename)
        return


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.5
    epsilon = 1
    tran_cost = 0
    num_states = 3
    train_set_size = 1000
    test_set_size = 1000
    output_folder = 'data/processed_data/latest_dataset'
    data = build_dataset.readDataFromCsv(output_folder)
    data.insert(num_states, 'time', range(data.shape[0]))
    data_train = data.iloc[:train_set_size][:]
    data_test = data.iloc[train_set_size:train_set_size + test_set_size][:]

    # initialize the model
    model = Env(num_states, epsilon, alpha, gamma, tran_cost)

    # train the model
    print('Starting initial model training --- ')
    model.train(data_train)
    print('--- Initial model training is complete')

    print('Starting model test --- ')
    # test the model
    # curr_state and next_state are set at model initialisation
    for t in range(1, data_test.shape[0]):
        new_observation = data_test.iloc[t:t+1]

        # calculate reward
        reward = new_observation.iat[0, model.next_state] + model.tran_cost * (model.next_state != model.curr_state)
        model.reward_hist.append(reward)
        model.total_reward += reward

        # update prediction track record
        opt_choice = int(new_observation.iloc[:, :-1].idxmax(axis=1))
        model.error.append(reward - new_observation.iloc[:,0:model.num_states].max(axis=1).item())
        model.timestep_state.append(model.next_state)

        # recommend next state
        model.predict(new_observation)

        # add new observation to KB and update centroids
        model.train(data_test.iloc[t-1:t+1])

    print('--- Model test is complete')
    print('Starting model output --- ')
    output_file = 'results/sim_results_' + str(time.datetime.now()) + '.xlsx'
    model.output(data_test, output_file)
    print('--- Model output is complete')




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
