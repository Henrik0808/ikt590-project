import random
import re
import json

import numpy as np

import config
import utils
from utils import get_categories


def read_records(file_name):
    with open(config.DATA_DIR + file_name, 'r') as file:
        data = json.load(file)

        return data['train'], data['val']


if __name__ == '__main__':
    PATTERN_SENTENCE_END = re.compile(r'[.!?]+')
    PATTERN_WHITESPACE = re.compile(r'\s+')
    # TODO: Add \\=$%&/`;- to PATTERN_NOISE? If so: PATTERN_NOISE = re.compile(r'[,"()[\]\\:^<=>$%&/`;*~_|#{}+-]+')
    PATTERN_NOISE = re.compile(r'[,"()[\]:^<>*~_|#{}+]+')

    records_train, records_val = read_records(config.FILE_CLINC150_ORIGINAL)

    queries_train = [q[0] for q in records_train]
    intents_train = [q[1] for q in records_train]

    queries_val = [q[0] for q in records_val]
    intents_val = [q[1] for q in records_val]

    # Replace whitespace with a single space
    queries_train = [PATTERN_WHITESPACE.sub(' ', q) for q in queries_train]
    # Remove any unwanted characters, including bounding whitespace
    queries_train = [PATTERN_NOISE.sub('', q).strip() for q in queries_train]
    # Remove sentence end signs
    queries_train = [PATTERN_SENTENCE_END.sub('', q).strip() for q in queries_train]

    # Replace whitespace with a single space
    queries_val = [PATTERN_WHITESPACE.sub(' ', q) for q in queries_val]
    # Remove any unwanted characters, including bounding whitespace
    queries_val = [PATTERN_NOISE.sub('', q).strip() for q in queries_val]
    # Remove sentence end signs
    queries_val = [PATTERN_SENTENCE_END.sub('', q).strip() for q in queries_val]

    queries_train_masked = []
    queries_train_masked_words = []
    queries_train_shuffled = []

    queries_val_masked = []
    queries_val_masked_words = []
    queries_val_shuffled = []

    for query in queries_train:
        random_word_index_masked = random.randint(0, len(query.split()) - 1)
        # Get masked word
        query_list = query.split()
        query_random_word_masked = query_list[random_word_index_masked]
        query_masked_word = query_list[:]
        query_masked_word[random_word_index_masked] = '[MASK]'

        query_shuffled = query_list[:]
        # X_shuffled: X (10 words) shuffled
        random.shuffle(query_shuffled)
        # X_shuffled_to_sentence_indexes = shuffled_words_to_sentence_indexes(X_shuffled, X)
        # X_shuffled_to_sentence_indexes = ' '.join(X_shuffled_to_sentence_indexes)
        query_masked_word = ' '.join(query_masked_word)
        query_shuffled = ' '.join(query_shuffled)

        queries_train_masked.append(query_masked_word)
        queries_train_masked_words.append(query_random_word_masked)
        queries_train_shuffled.append(query_shuffled)

    for query in queries_val:
        random_word_index_masked = random.randint(0, len(query.split()) - 1)
        # Get masked word
        query_list = query.split()
        query_random_word_masked = query_list[random_word_index_masked]
        query_masked_word = query_list[:]
        query_masked_word[random_word_index_masked] = '[MASK]'

        query_shuffled = query_list[:]
        # X_shuffled: X (10 words) shuffled
        random.shuffle(query_shuffled)
        # X_shuffled_to_sentence_indexes = shuffled_words_to_sentence_indexes(X_shuffled, X)
        # X_shuffled_to_sentence_indexes = ' '.join(X_shuffled_to_sentence_indexes)
        query_masked_word = ' '.join(query_masked_word)
        query_shuffled = ' '.join(query_shuffled)

        queries_val_masked.append(query_masked_word)
        queries_val_masked_words.append(query_random_word_masked)
        queries_val_shuffled.append(query_shuffled)

    # todo: include [MASK] here to make it part of tokenizer?
    queries_train = ['sos [MASK] ' + q for q in queries_train]
    queries_val = ['sos [MASK] ' + q for q in queries_val]

    for idx, intent in enumerate(intents_train):
        if intent == 'exchange_rate':
            intents_train[idx] = 'exchange_rate_clinc150'

    for idx, intent in enumerate(intents_val):
        if intent == 'exchange_rate':
            intents_val[idx] = 'exchange_rate_clinc150'

    records_train = list(zip(intents_train, queries_train,
                             queries_train_masked, queries_train_masked_words,
                             queries_train_shuffled))
    records_val = list(zip(intents_val, queries_val,
                           queries_val_masked, queries_val_masked_words,
                           queries_val_shuffled))

    categories = json.dumps(list(set(intents_val)))
    categories_parsed = json.loads(categories)

    with utils.cwd(config.DATA_DIR):
        print(f'Writing records to ... {config.DATA_DIR + config.FILE_TRAINING_CLINC150}')
        with open(config.FILE_TRAINING_CLINC150, mode='w', encoding='utf-8') as f:
            for record in records_train:
                f.write(','.join(record) + '\n')

    with utils.cwd(config.DATA_DIR):
        print(f'Writing records to ... {config.DATA_DIR + config.FILE_VALIDATION_CLINC150}')
        with open(config.FILE_VALIDATION_CLINC150, mode='w', encoding='utf-8') as f:
            for record in records_val:
                f.write(','.join(record) + '\n')

    with utils.cwd(config.DATA_DIR):
        print(f'Writing records to ... {config.DATA_DIR + config.FILE_CLINC150_CATEGORIES}')
        with open(config.FILE_CLINC150_CATEGORIES, mode='w', encoding='utf-8') as f:
            json.dump(categories_parsed, f, indent=2)
