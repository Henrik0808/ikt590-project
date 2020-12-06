TRAINING_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SIZE_OF_DATASET = 50_000

DATA_DIR = 'data/'
OUTPUTS_DIR = 'outputs/'
CHECKPOINTS_DIR = OUTPUTS_DIR + 'checkpoints/'

FILE_TRAINING = 'words_train.csv'
FILE_VALIDATION = 'words_valid.csv'
FILE_TESTING = 'words_test.csv'

SUPERVISED = 0
SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD = 1
SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS = 2
SEMI_SUPERVISED_PHASE_1_MASKED_WORD = 3
SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER = 4
SEMI_SUPERVISED_PHASE_2 = 5

# Model number 0: SimpleModel
# Model number 1: SimpleGRUModel
# Model number 2: Seq2seq
MODEL_NUMS = [0, 1, 2]

categories = {
    'religion': {'soc.religion.christian'},
    'computers': {'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'},
    'vehicles': {'rec.autos', 'rec.motorcycles'},
    'sports': {'rec.sport.baseball', 'rec.sport.hockey'},
    'science': {'sci.med', 'sci.space'}
}

VOCAB_SIZE = None
EMBED_DIM = 512
BATCH_SIZE = 512
N_FEATURES = 10
HIDDEN_DIM = 512
NUM_LAYERS = 1
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0
LEARNING_RATE = 0.001
TEACHER_FORCE_RATIO = 0.1

NUM_WORKERS = 0  # TODO: Fix bug where NUM_WORKERS >= 4 leads to WinError 1455

# Needed when using an encoder-decoder model,
# because the first 'word' input token to the decoder needs to be an sos (start of sequence) token
SOS_TOKEN_SUPERVISED = 0
SOS_TOKEN_SHUFFLED = N_FEATURES
SOS_TOKEN_VOCAB = 0

PAD_IDX = 0

TARGET_LEN = 1

FORCE_CPU = False

LOAD_MODEL = True
SAVE_MODEL = True

# Supervised (input: 10 words, output: category)
SUPERVISED_NUM_CLASSES = len(categories)
SUPERVISED_N_EPOCHS = 10

# Semi-supervised phase 1 (input: 10 words, output: 11th word)
SEMI_SUPERVISED_PHASE_1_N_EPOCHS = 10

# Semi-supervised phase 2 (input: 10 words, output: category)
SEMI_SUPERVISED_PHASE_2_NUM_CLASSES = SUPERVISED_NUM_CLASSES
SEMI_SUPERVISED_PHASE_2_N_EPOCHS = 10
