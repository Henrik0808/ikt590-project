import random
import re
import csv

import numpy as np

import config
import utils
from utils import get_categories


def read_records(file_name):
    records = []

    with open(config.DATA_DIR + file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            records.append(row)

    return records


if __name__ == '__main__':
    PATTERN_SENTENCE_END = re.compile(r'[.!?]+')
    PATTERN_WHITESPACE = re.compile(r'\s+')
    # TODO: Add \\=$%&/`;- to PATTERN_NOISE? If so: PATTERN_NOISE = re.compile(r'[,"()[\]\\:^<=>$%&/`;*~_|#{}+-]+')
    PATTERN_NOISE = re.compile(r'[,"()[\]:^<>*~_|#{}+]+')

    records_train = read_records(config.FILE_TRAINING_BANKING77_ORIGINAL)
    records_val = read_records(config.FILE_VALIDATION_BANKING77_ORIGINAL)

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

    # todo: include [MASK] here to make it part of tokenizer?
    queries_train = ['sos [MASK] ' + q for q in queries_train]
    queries_val = ['sos [MASK] ' + q for q in queries_val]

    for idx, intent in enumerate(intents_train):
        if intent == 'exchange_rate':
            intents_train[idx] = 'exchange_rate_banking77'

    for idx, intent in enumerate(intents_val):
        if intent == 'exchange_rate':
            intents_val[idx] = 'exchange_rate_banking77'

    records_train = list(zip(intents_train, queries_train))
    records_val = list(zip(intents_val, queries_val))

    with utils.cwd(config.DATA_DIR):
        print(f'Writing records to ... {config.DATA_DIR + config.FILE_TRAINING_BANKING77}')
        with open(config.FILE_TRAINING_BANKING77, mode='w', encoding='utf-8') as f:
            for record in records_train[1:]:
                f.write(','.join(record) + '\n')

    with utils.cwd(config.DATA_DIR):
        print(f'Writing records to ... {config.DATA_DIR + config.FILE_VALIDATION_BANKING77}')
        with open(config.FILE_VALIDATION_BANKING77, mode='w', encoding='utf-8') as f:
            for record in records_val[1:]:
                f.write(','.join(record) + '\n')
