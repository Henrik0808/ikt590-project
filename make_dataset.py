import os
import random
import re

import numpy as np
from sklearn.datasets import fetch_20newsgroups

import config

if __name__ == '__main__':
    categories = {
        'sports': {'rec.sport.baseball', 'rec.sport.hockey'},
        'religion': {'alt.atheism', 'soc.religion.christian', 'talk.religion.misc'},
        'computers': {'comp.graphics', 'comp.os.ms-windows.misc'},
    }

    categories_flat = []
    id2cat = {}

    idx = 0
    for category, boards in categories.items():
        for board in boards:
            categories_flat.append(board)
            id2cat[idx] = category
            idx += 1
    del idx
    print('Working on', len(categories), 'categories;',
          ', '.join(categories.keys()))

    # External fetch, might be cached
    print('Fetching dataset ...')
    messages, category_ids = fetch_20newsgroups(
        subset='all',
        categories=categories_flat,
        remove={'headers', 'footers', 'quotes'},
        return_X_y=True
    )
    print('Fetched dataset!')

    # Randomize the dataset order,
    # we don't know if it's fairly sorted
    random.seed(42)
    temp = list(zip(messages, category_ids))
    random.shuffle(temp)
    messages, category_ids = zip(*temp)
    del temp

    PATTERN_SENTENCE_END = re.compile(r'[.!?]+')
    PATTERN_WHITESPACE = re.compile(r'\s+')
    PATTERN_NOISE = re.compile(r'[,"()[\]:^<>*~_|#{}+]+')
    ignore_sequences = {'---', '==', '\\\\', '//', '@'}

    # Keep track category each data point is in
    metrics = {k: 0 for k in categories.keys()}

    data_points = []
    for message, category_id in zip(messages, category_ids):
        # Quit early when we don't need more data
        if len(data_points) >= config.SIZE_OF_DATASET:
            break

        # Split message into sentences
        sentences = PATTERN_SENTENCE_END.split(message)
        # Replace whitespace with a single space
        sentences = [PATTERN_WHITESPACE.sub(' ', x) for x in sentences]
        # Remove any unwanted characters, including bounding whitespace
        sentences = [PATTERN_NOISE.sub('', x).strip() for x in sentences]
        # For every non-empty sentence, lower all characters
        sentences = [x.lower() for x in sentences if x != '']

        for sentence in sentences:
            # Remove complete sentence if it contains certain sequences
            if any(sequence in sentence for sequence in ignore_sequences):
                continue

            words = sentence.split()

            # Skip sentences which are not at least 11 words long
            if not len(words) >= 11:
                continue

            # Iterate over words in a sliding window
            for head, tail in zip(range(len(words)-10), range(10, len(words))):
                if len(data_points) >= config.SIZE_OF_DATASET:
                    break

                # the 10 first words
                X = words[head:tail]
                X = ' '.join(X)
                # the 11th word
                Y = words[tail]

                data_points.append((id2cat[category_id], X, Y))
                metrics[id2cat[category_id]] += 1

    # Metrics
    print(f'Found a total of {len(data_points)} data points:')
    for category, hits in metrics.items():
        print('\tCategory:', category,
              '\tHits:', hits,
              f'({hits / len(data_points):.1%})')

    # Randomize the data points
    random.seed(42)
    random.shuffle(data_points)

    # Split for train, valid, and test datasets
    ratios = (.8, .1, .1)
    if not sum(ratios) == 1.:
        raise Exception('Splits must add up to 100%')

    split_data = np.split(data_points, [
        int(ratios[0] * len(data_points)),
        int((ratios[0] + ratios[1]) * len(data_points))
    ])

    # Write data points to file
    filenames = (config.FILE_TRAINING,
                    config.FILE_VALIDATION,
                    config.FILE_TESTING)

    os.chdir('data')
    for filename, data_points in zip(filenames, split_data):
        print(f'Writing data to ... {filename}')
        with open(filename, mode='w', encoding='utf-8') as f:
            for data_point in data_points:
                f.write(','.join(data_point) + '\n')
