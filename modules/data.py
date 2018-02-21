import csv
import modules.const as const
import numpy as np
import pandas as pd

# Read clean data from CSV and normalise
def read_data():
    df = pd.read_csv(const.FILE, index_col='Date')

    # normalise
    max_val = df.max()
    min_val = df.min()
    def normalise_data(row):
        for i, col in enumerate(df):
            row[col] = np.interp(row[col], [min_val[i], max_val[i]], [0, 1])
        return row

    df = df.apply(normalise_data, axis=1)
    return df
    # return np.array(df)

# Denormalise
def denormalise_data(val, max_val, min_val):
    return (val * (max_val - min_val)) + min_val






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
