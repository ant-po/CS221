import numpy as np
import pandas as pd
import build_dataset
import datetime as time

class Env():
    # Define class Env and assign appropriate parameters
    def __init__(self, num_states, tran_cost, decay, dist_check):
        self.num_states = num_states
        self.curr_state = 0
        self.next_state = np.random.randint(0, self.num_states)
        self.decay = decay
        self.dist_check = dist_check
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
        self.dist_hist = [0]
        return

    def train(self, data):
        # Create/update Knowledge Base and calculate centroids
        # observation : df (r1, r2, r3, time)
        # KB : dict(key: current state, val: dict(key: opt next state, val: df(observations)))
        prior_observation = data.iloc[0:1]
        for t in range(1, data.shape[0]):
            new_observation = data.iloc[t:t + 1]
            for i in range(self.num_states):
                observation_after_cost = new_observation.iloc[:, :-1] + self.tran_cost
                observation_after_cost.iloc[:][str(i)] -= self.tran_cost
                opt_choice = int(observation_after_cost.idxmax(axis=1))
                if opt_choice in self.KB[i].keys():
                    self.KB[i][opt_choice] = self.KB[i][opt_choice].append(prior_observation, ignore_index=True)
                else:
                    self.KB[i][opt_choice] = prior_observation
            prior_observation = new_observation
        self.updateCentroids()
        return

    def updateCentroids(self):
        # update centroids given the latest Knowledge Base
        for i in range(self.num_states):
            for j in range(self.num_states):
                data_slice = self.KB[i][j]
                if self.decay:
                    x = data_slice.iloc[:, :-1].mul(data_slice.loc[:, 'time'], axis=0).sum(
                        axis=0) / data_slice.loc[:, 'time'].sum()
                else:
                    x = data_slice.iloc[:, :-1].sum(axis=0) / data_slice.shape[0]
                self.centroids[i][j] = pd.DataFrame(x).transpose()

    def predict(self, observation):
        # make a prediction based on the distance between the centroid and observation
        self.curr_state = self.next_state
        # find the distance between centroid and the observation
        distances = self.distToCentroid(observation)
        if self.dist_check:
            if np.mean(distances) < np.percentile(self.dist_hist, 75):
                self.next_state = self.curr_state
            else:
                self.next_state = np.argmin(distances)
        else:
            self.next_state = np.argmin(distances)
        self.dist_hist.append(np.min(distances))
            # self.next_state = int(np.random.choice(self.num_states, 1, p=distances.flatten()/sum(distances)))

        return

    def distToCentroid(self, observation):
        # Calculate euclidean distance between the observation and relevant centroid
        distances = []
        xyz = [str(i) for i in range(self.num_states)]
        for i in range(self.num_states):
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
    tran_cost = -0.0001
    decay = False
    dist_check = False
    num_states = 3
    train_set_size = 260
    test_set_size = 5000
    output_folder = 'data/processed_data/latest_dataset'
    data = build_dataset.readDataFromCsv(output_folder)
    data.insert(num_states, 'time', range(data.shape[0]))
    data_train = data.iloc[:train_set_size][:]
    data_test = data.iloc[train_set_size:min(train_set_size+test_set_size, data.shape[0])][:]

    # initialize the model
    model = Env(num_states, tran_cost, decay, dist_check)

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
        model.error.append(reward - new_observation.iloc[:, 0:model.num_states].max(axis=1).item())
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
