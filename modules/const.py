# Data
FILE = './data/data_clean.csv'
FEATURES = 5    # predictors
COLUMNS = ['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']

TRAIN_SPLIT = 0.7

# Model
NEURONS = [FEATURES, 4, 1] # Input, Hidden neurons, Output neurons
ACTIVATION = 'Sigmoid'
LEARNING_RATE = 0.1

RANDOM_SEED = 1
