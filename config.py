import json

TRAINING_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SIZE_OF_DATASET = 500_000

DATA_DIR = 'data/'
OUTPUTS_DIR = 'outputs/'
CHECKPOINTS_DIR = OUTPUTS_DIR + 'checkpoints/'

# Changed category "exchange_rate" to "exchange_rate_banking77",
# in order to avoid category name overlap in cat2id dict
FILE_BANKING77_CATEGORIES = 'banking77_categories.json'
FILE_CLINC150_CATEGORIES = 'clinc150_categories.json'  # Changed category "exchange_rate" to "exchange_rate_clinc150"
FILE_CLINC150_ORIGINAL = "clinc150_data_full.json"
FILE_TRAINING_BANKING77_ORIGINAL = 'banking77_train_original.csv'
FILE_TRAINING_BANKING77 = 'banking77_train.csv'
FILE_TRAINING_CLINC150 = 'clinc150_train.csv'
FILE_TRAINING_CLINC150_RNN = 'clinc150_train_rnn.csv'
FILE_VALIDATION_BANKING77_ORIGINAL = 'banking77_valid_original.csv'
FILE_VALIDATION_BANKING77 = 'banking77_valid.csv'
FILE_VALIDATION_CLINC150 = 'clinc150_valid.csv'
FILE_VALIDATION_CLINC150_RNN = 'clinc150_valid_rnn.csv'

FILE_TRAINING = 'words_train.csv'
FILE_TRAINING_RNN = 'words_train_rnn.csv'
FILE_VALIDATION = 'words_valid.csv'
FILE_VALIDATION_RNN = 'words_valid_rnn.csv'
FILE_TESTING = 'words_test.csv'
FILE_TESTING_RNN = 'words_test_rnn.csv'

SEMI_SUPERVISED = None
USING_SIMPLE_MODEL = None

SUPERVISED_BANKING77 = 0
SUPERVISED_20NEWS = 1
SUPERVISED_CLINC150 = 2
SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_BANKING77 = 3
SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS = 4
SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77 = 5
SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS = 6
SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150 = 7
SEMI_SUPERVISED_PHASE_1_MASKED_WORD_BANKING77 = 8
SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS = 9
SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 = 10
SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77 = 11
SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS = 12
SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 = 13
SEMI_SUPERVISED_PHASE_2_BANKING77 = 14
SEMI_SUPERVISED_PHASE_2_20NEWS = 15
SEMI_SUPERVISED_PHASE_2_CLINC150 = 16

PREPROC_MAP = {0: 'supervised banking77', 1: 'eleventh', 2: 'shuffled', 3: 'masked', 4: 'autoenc',
               9: 'masked 20news', 10: 'masked clinc150', 12: 'autoenc 20news', 13: 'autoenc clinc150',
               14: 'downstream banking77'}

MODEL_MAP = {0: 'simple', 1: 'simplegru', 2: 'seq2seq'}

SUPERVISED = None
PHASE_1 = None
PHASE_2 = None
NUM_CLASSES = None

MAX_QUERY_LEN = None

# Model number 0: SimpleModel
# Model number 1: SimpleGRUModel
# Model number 2: Seq2seq
MODEL_NUMS = [0, 1, 2]

categories_20news = {
    'religion': {'soc.religion.christian'},
    'computers': {'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'},
    'vehicles': {'rec.autos', 'rec.motorcycles'},
    'sports': {'rec.sport.baseball', 'rec.sport.hockey'},
    'science': {'sci.med', 'sci.space', 'sci.electronics'},
    'politics': {'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'},
    'sale': {'misc.forsale'}
}

with open(DATA_DIR + FILE_BANKING77_CATEGORIES) as json_file:
    categories_banking77 = json.load(json_file)

with open(DATA_DIR + FILE_CLINC150_CATEGORIES) as json_file:
    categories_clinc150 = json.load(json_file)

VOCAB_SIZE = None
EMBED_DIM = 512
BATCH_SIZE = 512
N_FEATURES = 10
HIDDEN_DIM = 1024
NUM_LAYERS = 1
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0
LEARNING_RATE = 0.0001
TEACHER_FORCE_RATIO = 0.1

NUM_WORKERS = 0  # TODO: Fix bug where NUM_WORKERS >= 4 leads to WinError 1455

# Needed when using an encoder-decoder model,
# because the first 'word' input token to the decoder needs to be an sos (start of sequence) token
SOS_TOKEN_SUPERVISED = 0
SOS_TOKEN_SHUFFLED = N_FEATURES
SOS_TOKEN_VOCAB = 0

PAD_IDX = 0

DEVICE = None

TOKENIZER = None
cat2id = None

TARGET_LEN = None
TARGET_LEN_MASKED = 2

FORCE_CPU = False

SAVE_MODEL = True
CONTINUE_TRAINING_MODEL = False

N_EPOCHS = 200

# Supervised (input: 10 words, output: category)
SUPERVISED_NUM_CLASSES_20NEWS = len(categories_20news)
SUPERVISED_NUM_CLASSES_BANKING77 = len(categories_banking77)
SUPERVISED_N_EPOCHS = N_EPOCHS

# Semi-supervised phase 1 (input: 10 words, output: 11th word)
SEMI_SUPERVISED_PHASE_1_N_EPOCHS = N_EPOCHS

# Semi-supervised phase 2 (input: 10 words, output: category)
SEMI_SUPERVISED_PHASE_2_NUM_CLASSES_20news = SUPERVISED_NUM_CLASSES_20NEWS
SEMI_SUPERVISED_PHASE_2_NUM_CLASSES_banking77 = SUPERVISED_NUM_CLASSES_BANKING77
SEMI_SUPERVISED_PHASE_2_N_EPOCHS = N_EPOCHS
