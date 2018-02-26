# Data
FILE = './data/data_clean.csv'
FEATURES = 5    # predictors
COLUMNS = ['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']

TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.2

# Model
NEURONS = [FEATURES, 4, 1] # Input, Hidden neurons, Output neurons
ACTIVATION = 'Sigmoid'
LEARNING_RATE = 0.1
MAX_EPOCHS = 5000
VALIDATION_EPOCHS = 50 # check validation set error every X epochs

RANDOM_SEED = 50
