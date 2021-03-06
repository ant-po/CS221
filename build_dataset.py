"""
This set of functions is responsible for sourcing and processing data
Data parameter file is in ./data/data_params
Raw data is saved in ./data/raw_data folder
Processed data is saved in ./data/processed_data
In order to use processed data for learning, drop the files in
./data/processed_data/latest_dataset
"""

import argparse
import datetime
import os
import time
import fix_yahoo_finance as fix
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
from utils import Params
from itertools import permutations

parser = argparse.ArgumentParser()
parser.add_argument('--data_params', default='data/data_params', help="Directory with the params.json")
parser.add_argument('--data_dir', default='data/raw_data', help="Directory with the raw price data")
parser.add_argument('--output_dir', default='data/processed_data', help="Where to write the new data")


def fetchData(dataCodes, startDate, endDate, output_folder):
    """
    Gets historical stock data of given tickers between dates
    :param dataCode: security (securities) whose data is to fetched
    :type dataCode: string or list of strings
    :param startDate: start date
    :type startDate: string of date "YYYY-mm-dd"
    :param endDate: end date
    :type endDate: string of date "YYYY-mm-dd"
    :return: saves data in a csv file with the timestamps
    """
    fix.pdr_override()
    data = {}
    # for code in dataCodes:
    i = 1
    try:
        all_data = pdr.get_data_yahoo(dataCodes, startDate, endDate)
    except ValueError:
        print("ValueError, trying again")
        i += 1
        if i < 5:
            time.sleep(10)
            fetchData(dataCodes, startDate, endDate)
        else:
            print("Tried 5 times, Yahoo error. Trying after 2 minutes")
            time.sleep(120)
            fetchData(dataCodes, startDate, endDate)
    all_data = all_data.fillna(method="ffill")
    all_data = all_data.fillna(method="bfill")
    data = all_data["Adj Close"]
    # output the results in a csv file
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')
    filename = output_folder + "/raw_data_" + time_stamp + ".csv"
    data.columns = [i for i in range(len(dataCodes))]
    data.index = range(len(data))
    data.to_csv(filename)
    print("Data has been saved in a CSV format in ", filename)
    return data


def getExistingData(filename):
    """Find existing data file "filename" and extract the data"""
    data = pd.read_csv(filename)
    return data

def saveDataToCsv(data, output_folder):
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')
    output_folder = output_folder + "/data_set_" + time_stamp
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data.to_csv(output_folder + '/state_data.csv')
    print("Data has been saved in a CSV format in ", output_folder)

def readDataFromCsv(output_folder):
    """Import the training/test data from CSV files to Numpy arrays"""
    data = pd.read_csv(output_folder + '/state_data.csv', index_col=0)
    return data

def rankToLabel(rank):
    """Generate a label array of size [rank.size**2,1] from the rank
    e.g. rank = [1,2] --> label = [1, 0, 0, 1]"""
    num_examples, num_assets = rank.shape
    label = np.zeros((num_examples, num_assets ** 2))
    row = 0
    for elem in rank:
        col = 0
        for el in elem:
            col += 1
            label[row, (col-1)*num_assets+int(el)-1] = 1
        row += 1
    return label


def labelToRank(label):
    """Generate a rank array of size [sqrt(label.size),1] from the label
        e.g. label = [1, 0, 0, 1] --> ranking = [1,2]"""
    num_examples, num_assets = label.shape
    num_assets = np.int(np.sqrt(num_assets))
    rank = np.zeros((num_examples, num_assets))
    for j in range(0, num_examples):
        for i in range(1, num_assets**2+1):
            if i % num_assets == 0:
                rank[j, int(i/num_assets)-1] = list(label[j, i-num_assets:i]).index(1)+1
    return rank


def logitToLabel(logit):
    """Generate labels from logits based on max value"""
    num_examples, num_assets_sq = logit.shape
    label = np.zeros((num_examples, num_assets_sq))
    num_assets = np.int(np.sqrt(num_assets_sq))
    for j in range(0, num_examples):
        for i in range(1, num_assets_sq+1):
            if i % num_assets == 0:
                index = np.argmax(logit[j, i-num_assets:i])
                label[j, (int(i/num_assets)-1)*num_assets+index] = 1
    return label


def rankToPermLabel(rank_vec):
    """Convert a rank label to a label based on classes corresponding
        to all unique permutations of the rank label
        e.g. 2 assets --> PermLabel = [1, 2], [2, 1]
        rank_vec = [1, 2] --> rankToPermLabel = [1, 0]"""
    if len(rank_vec.shape) == 1:
        perm_label = np.zeros((1, np.math.factorial(rank_vec.shape[0])))
        num = rank_vec.shape[0]
        loop = False
    else:
        perm_label = np.zeros((rank_vec.shape[0], np.math.factorial(rank_vec.shape[1])))
        num = rank_vec.shape[1]
        loop = True
    list_perm = np.array(list(permutations(range(1, num + 1))))
    row = 0
    if loop:
        for rank in rank_vec:
            index = np.where((list_perm==rank).all(-1))[0][0]
            perm_label[row, index] = 1
            row += 1
    else:
        index = np.where((list_perm == rank_vec).all(-1))[0][0]
        perm_label[row, index] = 1
    return perm_label


def processData(rets):
    """"convert daily returns to [mean, std, max, min] features"""""
    stats = np.c_[np.mean(rets, axis=1), np.std(rets, axis=1), np.max(rets, axis=1), np.min(rets, axis=1)]
    return stats


def pickData(rets, freq):
    """sample daily returns at a particular frequency"""
    out = rets[range(0, rets.shape[0], freq), :]
    return out


def pickBest(label):
    """converte rank to label pointing to the position of max rank"""
    out = np.zeros(label.shape)
    out[label == 3] = 1
    return out


def staggered_sum(x, w):
    """calculate returns over staggered windows of size w"""
    count = 0
    out = np.zeros((np.int(x.shape[0]/w), x.shape[1]))
    for i in range(0, x.shape[0]-w+1, w):
        out[count, :] = np.sum(x[i:i+w+1, :], axis=0)
        count += 1
    return out


def getXYSet(data, look_back, invest_horizon):
    """Slice the data to create training X,Y examples"""
    time_series = data.values
    return time_series


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_data')
    test_data_dir = os.path.join(args.data_dir, 'test_data')

    # Read data params config
    json_path = os.path.join(args.data_params, 'data_params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    data_params = Params(json_path)

    # Fetch data
    hist_prices = fetchData(data_params.dataCodes, data_params.startDate, data_params.endDate, args.data_dir)

    # Convert prices to log returns
    hist_returns = hist_prices.pct_change(1)
    hist_returns.fillna(0, inplace=True)

    # Output the results to csv
    saveDataToCsv(hist_returns, args.output_dir)
    print("Done building dataset. Ready to train the model now!")
