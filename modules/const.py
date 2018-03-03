# Data
FILE = './data/data_clean.csv'
FEATURES = 5    # predictors
COLUMNS = ['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']

TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.2

# Model
NEURONS = [FEATURES, 100, 1] # Input, Hidden neurons, Output neurons
LEARNING_RATE = 0.1
MOMENTUM_ALPHA = 0.9
MOMENTUM = True
BOLD_DRIVER = False
BOLD_DRIVER_INCREASE = 1.1  # Multiply LR by this amount if error is lower
BOLD_DRIVER_DECREASE = 0.5
MAX_EPOCHS = 1000
VALIDATION_EPOCHS = 100 # check validation set error every X epochs

# Options
TRAIN = True
MODEL_DIR = './model/'
MODEL_NAME = 'rXX'   # save model
LOAD_MODEL = ''   # '' or model name

RANDOM_SEED = 50
