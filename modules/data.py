import csv
import modules.const as const
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(const.RANDOM_SEED)
plt.rcParams['figure.figsize'] = [16.0, 10.0]

# Read clean data from CSV, normalise, and split
def read_data():
    df = pd.read_csv(const.FILE, index_col='Date')
    df = df.drop('Month', 1)

    # normalise
    max_val = df.max()
    min_val = df.min()
    def normalise_data(row):
        for i, col in enumerate(df):
            row[col] = np.interp(row[col], [min_val[i], max_val[i]], [0, 1])
        return row

    df = df.apply(normalise_data, axis=1)
    data = np.array(df)

    # shuffle data
    np.random.shuffle(data)

    train_size = int(const.TRAIN_SPLIT * len(data))

    train_x = data[:train_size][:, :-1]
    train_y = data[:train_size][:, -1:]
    test_x = data[train_size:][:, :-1]
    test_y = data[train_size:][:, -1:]

    return train_x, train_y, test_x, test_y

# Denormalise
def denormalise_data(val, max_val, min_val):
    return (val * (max_val - min_val)) + min_val

# Plot prediction data
def plot(pred, expected):
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    ax1.grid(True)

    ax1.plot(pred, 'x', markersize=7, label='predicted', color='r')
    ax1.plot(expected, 'o', markersize=7, label='expected', color='b')

    plt.xlabel('Data point')
    plt.ylabel('PanE')
    plt.title('PanE Prediction')

    plt.legend()
    plt.subplots_adjust(left=0.06, bottom=0.18, right=0.95, top=0.95, wspace=0.2, hspace=0)
    plt.show()




#############################
# Without using Pandas/Numpy:

# Read clean data from CSV file
def __read_data():
    with open(const.FILE, 'rb') as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=',', quotechar='\'')
        for r in reader:
            data.append({'date':r[0], 'month':r[1], 'T':r[2], 'W':r[3], 'SR':r[4], 'DSP':r[5], 'DRH':r[6], 'PanE':r[7]})
        return data[1:] # don't return header row

# Normalise data
def __normalise_data(data):
    for col in const.COLUMNS:
        coldata = [float(row[col]) for row in data]
        colmin = min(coldata)
        colmax = max(coldata)

        colnorm = [((val - colmin)/(colmax - colmin)) * (const.NORM_MAX - const.NORM_MIN) + const.NORM_MIN for val in coldata]

        for idx, row in enumerate(data):
            data[idx][col] = colnorm[idx]

    return data
