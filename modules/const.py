# Data
FILE = './data/data_clean.csv'
FEATURES = 5    # predictors
COLUMNS = ['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']

TRAIN_SPLIT = 0.6

# Model
NEURONS = [FEATURES, 2, 1] # Input, Hidden neurons, Output neurons
ACTIVATION = 'Sigmoid'
LEARNING_RATE = 0.1
