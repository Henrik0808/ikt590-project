import os
import random
import re

from sklearn.datasets import fetch_20newsgroups

import config


# Create file containing training, validation or testing dataset
def create_dataset_file(file, dataset, category_id_to_name):
    with open(file, mode='w', encoding='utf-8') as f:
        for X, Y, eleventh_word in dataset:
            f.write(category_id_to_name(Y) + ',' + X + ',' + eleventh_word + '\n')


def main():
    if not os.path.isdir('data'):
        os.mkdir('data')

    categories = ('rec.sport.baseball', 'rec.sport.hockey',
                  'alt.atheism', 'soc.religion.christian', 'talk.religion.misc',
                  'comp.graphics', 'comp.os.ms-windows.misc')

    category_id_to_name = lambda x: 'sports' if x < 2 else ('religion' if x < 5 else 'computers')

    Xs, Ys = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        return_X_y=True
    )

    PATTERN_SENTENCE_END = re.compile(r'[.!?]+')
    PATTERN_WHITESPACE = re.compile(r'\s+')
    PATTERN_NOISE = re.compile(r'[,"()[\]:^<>*~_|#{}+]+')
    ignore_sequences = ('---', '==', '\\\\', '//', '@')

    clean_10 = []

    for X, Y in zip(Xs, Ys):
        sentences = PATTERN_SENTENCE_END.split(X)
        sentences = [PATTERN_NOISE.sub('', PATTERN_WHITESPACE.sub(' ', sentence)).strip() for sentence in sentences]
        sentences = [sentence.lower() for sentence in sentences if sentence != '']

        for sentence in sentences:
            if any(sequence in sentence for sequence in ignore_sequences):
                continue

            word_count = len(sentence.split())

            if word_count >= 11:  # Only sentences with at least 11 words are used for creating the dataset
                words = sentence.split()

                # Left side of moving 'window',
                window_left_index = 0

                # Right side of moving 'window'
                window_right_index = 10

                while True:
                    # Get 10 words from sentence within the 'moving window'
                    ten_word_sentence = words[window_left_index:window_right_index]
                    ten_word_sentence_full = ' '.join(ten_word_sentence)

                    clean_10.append((ten_word_sentence_full, Y, words[window_right_index]))

                    # Move window to the right one step within the sentence
                    window_left_index += 1
                    window_right_index += 1

                    # If the window has reached the last 11-word part of the sentence, break out of loop
                    if window_right_index == len(words):
                        break

    random.shuffle(clean_10)

    clean_10 = clean_10[:config.SIZE_OF_DATASET]

    sum1, sum2, sum3 = 0, 0, 0

    for _, Y, eleventh_word in clean_10:
        cat = category_id_to_name(Y)

        if cat == 'sports':
            sum1 += 1
        elif cat == 'religion':
            sum2 += 1
        else:
            sum3 += 1

    print(f'words_10.csv, sports: {sum1}, religion: {sum2}, computers: {sum3}')

    train_idx = int(len(clean_10) * config.TRAINING_RATIO)
    val_idx = int(len(clean_10) * config.VAL_RATIO + train_idx)

    # Split total dataset (clean_10) into training, validation and testing datasets
    words_train = clean_10[:train_idx]
    words_val = clean_10[train_idx:val_idx]
    words_test = clean_10[val_idx:]

    # Create file containing training dataset
    create_dataset_file(os.path.join(config.DATA_DIR, config.FILE_TRAINING), words_train, category_id_to_name)

    # Create file containing validation dataset
    create_dataset_file(os.path.join(config.DATA_DIR, config.FILE_VALIDATION), words_val, category_id_to_name)

    # Create file containing testing dataset
    create_dataset_file(os.path.join(config.DATA_DIR, config.FILE_TESTING), words_test, category_id_to_name)


if __name__ == '__main__':
    main()
