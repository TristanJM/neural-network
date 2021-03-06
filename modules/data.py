import warnings
import modules.const as const
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(const.RANDOM_SEED)
plt.rcParams['figure.figsize'] = [16.0, 10.0]
warnings.simplefilter(action='ignore', category=FutureWarning)  # suppress np/mpl FutureWarning

col_max = None
col_min = None

# Read clean data from CSV, normalise, and split
def read_data():
    df = pd.read_csv(const.FILE, index_col='Date')
    df = df.drop('Month', 1)

    # normalise
    global col_max
    global col_min
    col_max = df.max()
    col_min = df.min()
    def normalise_data(row):
        for i, col in enumerate(df):
            row[col] = np.interp(row[col], [col_min[i], col_max[i]], [0, 1])
        return row

    df = df.apply(normalise_data, axis=1)
    data = np.array(df)

    # shuffle data
    np.random.shuffle(data)

    train_size = int(const.TRAIN_SPLIT * len(data))
    val_size = int(const.VALIDATION_SPLIT * len(data))

    train_x = data[:train_size][:, :-1]
    train_y = data[:train_size][:, -1:]
    val_x = data[train_size:train_size+val_size][:, :-1]
    val_y = data[train_size:train_size+val_size][:, -1:]
    test_x = data[train_size+val_size:][:, :-1]
    test_y = data[train_size+val_size:][:, -1:]

    return train_x, train_y, val_x, val_y, test_x, test_y

# Denormalise
def denormalise_data(val, dn_all=False):
    if dn_all:
        for i, row in enumerate(val):
            for j, col in enumerate(row):
                max_val = col_max[j]
                min_val = col_min[j]
                row[j] = (col * (max_val - min_val)) + min_val
        return val
    else:
        max_val = col_max['PanE']
        min_val = col_min['PanE']
        return (val * (max_val - min_val)) + min_val

# Plot prediction data
def plot(pred, expected, plot_type='scatter'):
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    ax1.grid(True)

    if plot_type == 'line':
        ax1.plot(pred, 'x', markersize=7, label='predicted', color='r')
        ax1.plot(expected, 'o', markersize=7, label='expected', color='b')
        plt.xlabel('Data point')
        plt.ylabel('PanE')
    else:
        max_val = col_max['PanE']
        min_val = col_min['PanE']
        ax1.plot([min_val,max_val],[min_val,max_val], color='black', alpha=0.5, label='Expected')   # reference line
        ax1.scatter(expected, pred, label='PanE')
        plt.xlabel('Expected')
        plt.ylabel('Predicted')

    plt.title('PanE Prediction')
    plt.legend()
    plt.subplots_adjust(left=0.06, bottom=0.18, right=0.95, top=0.95, wspace=0.2, hspace=0)
    plt.show()
