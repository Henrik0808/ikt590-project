TRAINING_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SIZE_OF_DATASET = 375_000

DATA_DIR = 'data/'

FILE_TRAINING = 'words_train.csv'
FILE_VALIDATION = 'words_valid.csv'
FILE_TESTING = 'words_test.csv'

SUPERVISED = 0
SEMI_SUPERVISED_PHASE_1 = 1
SEMI_SUPERVISED_PHASE_2 = 2

SUPERVISED_CLASSES = ('religion', 'computers', 'vehicles', 'sports', 'science')

VOCAB_SIZE = 0
EMBED_DIM = 512
BATCH_SIZE = 512
N_FEATURES = 10
HIDDEN_DIM = 512

NUM_WORKERS = 0  # TODO: Fix bug where NUM_WORKERS >= 4 leads to Winerror 1455

# Supervised (input: 10 words, output: category)
SUPERVISED_NUM_CLASSES = len(SUPERVISED_CLASSES)
SUPERVISED_N_EPOCHS = 30

# Semi-supervised phase 1 (input: 10 words, output: 11th word)
SEMI_SUPERVISED_PHASE_1_N_EPOCHS = 30

# Semi-supervised phase 2 (input: 10 words, output: category)
SEMI_SUPERVISED_PHASE_2_NUM_CLASSES = SUPERVISED_NUM_CLASSES
SEMI_SUPERVISED_PHASE_2_N_EPOCHS = 30
