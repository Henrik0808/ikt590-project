import os
import pathlib
import random
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from keras_preprocessing import text
from keras_preprocessing import sequence
from torch.utils.data import DataLoader, Dataset

import config
import training_testing
import utils
import plot_viewer
import copy


class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order

        # Make copy of batch, in order to not change the original dataset permanently
        batch_copy = copy.deepcopy(batch)

        if config.SEMI_SUPERVISED == config.SUPERVISED_BANKING77 or \
                config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_2_BANKING77:
            query = 'query'
            label = 'category'
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS:
            query = 'query'
            label = 'eleventh word'
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150:
            query = 'query'
            label = 'masked word'
            # todo: only do this if batch is from train data not val data?
            masked_token = config.TOKENIZER.word_index['[mask]']
            # for each 'query': mask 1 random word
            for d in batch_copy:
                len_d = len(d[query])
                #num_words_masked = int(len_d * 0.15)

                #if num_words_masked == 0:
                    #num_words_masked = 1

                random_idxs = []

                for idx in range(config.TARGET_LEN_MASKED):
                    rand_idx = random.randint(0, len_d - 1)

                    if len_d > config.TARGET_LEN_MASKED - 1:
                        while rand_idx in random_idxs:
                            rand_idx = random.randint(0, len_d - 1)

                    random_idxs.append(rand_idx)

                random_idxs.sort()

                masked_words = []

                for random_idx in random_idxs:
                    masked_word = d[query][random_idx]
                    rand_num = random.random()

                    if rand_num < 2:
                        d[query][random_idx] = masked_token
                    else:
                        random_vocab_int = random.randint(4, config.VOCAB_SIZE - 1)
                        d[query][random_idx] = random_vocab_int
                    #else:
                        # Unchanged token
                        #masked_words.append(np.int64(0))
                        #continue

                    masked_words.append(masked_word)

                d[label] = np.asarray(masked_words)

        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS:
            query = 'query'
            label = 'query'
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS:
            query = 'query shuffled'
            label = 'query'
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
            query = 'query'
            label = 'query'
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
            query = 'shuffled query'
            label = 'query'

        sorted_batch = sorted(batch_copy, key=lambda x: x[query].shape[0], reverse=True)

        # Get each sequence and pad it
        sequences = [x[query] for x in sorted_batch]

        sequences = [torch.Tensor(t) for t in sequences]

        if isinstance(config.MODEL, training_testing.SimpleModel):

            sequences_padded = sequence.pad_sequences(sequences, maxlen=config.MAX_QUERY_LEN,
                                                      dtype=np.int64, padding='post',
                                                      truncating='post')

            sequences_padded = torch.from_numpy(sequences_padded).to(config.DEVICE)
        else:
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(config.DEVICE)  # todo: all torch to device?

        if config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:  # todo: what about 20news?
            # Get each shuffled sequence and pad it
            sequences_shuffled = [x[query] for x in sorted_batch]

            sequences_shuffled = [torch.Tensor(t).to(config.DEVICE) for t in sequences_shuffled]

            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences_shuffled, batch_first=True).to(config.DEVICE)

        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences]).to(config.DEVICE)

        if config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150:
            # Don't forget to grab the labels of the *sorted* batch
            labels = sequences_padded
        elif config.SEMI_SUPERVISED == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
            # Get each shuffled sequence and pad it
            queries = [x[label] for x in sorted_batch]

            queries = [torch.Tensor(t).to(config.DEVICE) for t in queries]

            queries_padded = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True).to(config.DEVICE)

            labels = queries_padded
        else:
            labels = torch.LongTensor([c[label] for c in sorted_batch]).to(config.DEVICE)

        return sequences_padded, lengths, labels


class WordsDataset(Dataset):
    """Words dataset containing training, validation or testing data."""

    # The tokenizer is created from the 10-word sentences in the training dataset.
    # The same tokenizer is used in all three dataset instances, to use the same indexes/tokens for the same words etc.
    tokenizer = None

    @staticmethod
    def create_tokenizer(records, records_other_train):
        # Unknown words will have index 1
        tokenizer = text.Tokenizer(oov_token=1, filters='')

        if config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS or \
                config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_ELEVENTH_WORD_20NEWS:
            # Join 10 word sentence and 11th word by space
            records = [' '.join(record) for record in records[:, 1:3]]
        elif config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 or \
                config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
                config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
            records = [r[1] for r in records]

        records_other_train = [r[1] for r in records_other_train]
        records_tot = records + records_other_train
        tokenizer.fit_on_texts(records_tot)
        # Limit to words used at least twice
        count_1 = sum(count == 1 for count in tokenizer.word_counts.values())
        # Adding 2 to num_words because (padding) index 0 and (unknown) index 1 also needs to be included
        tokenizer.num_words = len(tokenizer.word_counts) - count_1 + 2
        # Adding 1 to vocabulary size because of additional reserved padding index 0
        config.VOCAB_SIZE = len(tokenizer.word_index) + 1 - count_1
        # Get sos token, which is needed when using an encoder-decoder model
        config.SOS_TOKEN_VOCAB = tokenizer.word_index['sos']
        config.TOKENIZER = tokenizer  # For testing purposes
        return tokenizer

    @staticmethod
    def read_records(dataset_file):
        with open(dataset_file, mode='r', encoding='utf-8') as f:
            return np.array([
                line.split(',') for line in
                (line.rstrip() for line in f.readlines())  # todo: remove [:100] when done testing
            ])

    def __init__(self, dataset_file, transform=None):
        """
        Args:
            dataset_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample. Probably not needed here.
        """
        records = self.__class__.read_records(dataset_file)
        self.len = len(records)

        if not self.__class__.tokenizer:
            if config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_BANKING77 or \
                    config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_BANKING77 or \
                    config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_BANKING77:
                records_first_dataset = self.__class__.read_records(config.DATA_DIR + config.FILE_TRAINING_BANKING77)
            elif config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS or \
                    config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_20NEWS or \
                    config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_20NEWS:
                records_first_dataset = self.__class__.read_records(config.DATA_DIR + config.FILE_TRAINING)
            elif config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_CLINC150 or \
                    config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_AUTO_ENCODER_CLINC150 or \
                    config.PHASE_1 == config.SEMI_SUPERVISED_PHASE_1_SHUFFLED_WORDS_CLINC150:
                records_first_dataset = self.__class__.read_records(config.DATA_DIR + config.FILE_TRAINING_CLINC150)

            if config.PHASE_2 == config.SEMI_SUPERVISED_PHASE_2_BANKING77:
                records_second_dataset = self.__class__.read_records(config.DATA_DIR + config.FILE_TRAINING_BANKING77)
            elif config.PHASE_2 == config.SEMI_SUPERVISED_PHASE_2_20NEWS:
                records_second_dataset = self.__class__.read_records(config.DATA_DIR + config.FILE_TRAINING)
            elif config.PHASE_2 == config.SEMI_SUPERVISED_PHASE_2_CLINC150:
                records_second_dataset = self.__class__.read_records(config.DATA_DIR + config.FILE_TRAINING_CLINC150)

            self.len_first_dataset = len(records_first_dataset)
            self.len_second_dataset = len(records_second_dataset)

            self.__class__.tokenizer = \
                self.__class__.create_tokenizer(records_first_dataset, records_second_dataset)

        self.records_tokenized = self.tokenize(records)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.records_tokenized.items()}

        if self.transform:
            return self.transform(sample)

        if 'query' in sample:
            # Remove the sos token before the start of the 10 word sentence
            sample['query'] = sample['query'][2:]

        return sample

    def tokenize(self, records):
        _, cat2id, _ = utils.get_categories()
        config.cat2id = cat2id

        if cat2id[records[0, 0]] < 7:
            tokenized = np.asarray(self.__class__.tokenizer.texts_to_sequences(
                ' '.join(record) for record in records[:, 1:]
            ), dtype=np.int64)

            return {
                'category': [cat2id[record] for record in records[:, 0]],
                'query': tokenized[:, :12],
                'eleventh word': tokenized[:, 12]
            }
        elif cat2id[records[0, 0]] < 84:
            # Convert records[:, 1] (all queries) to array of arrays
            tokenized = self.__class__.tokenizer.texts_to_sequences(records[:, 1])
            tokenized = np.array([np.array(q, np.int64) for q in tokenized])

            return {
                # -5 because of targets shifted 5 to right side 5->0, 6->1 etc
                'category': [cat2id[record] - 7 for record in records[:, 0]],
                'query': tokenized
            }
        else:
            # Convert records[:, 1] (all queries) to array of arrays
            tokenized_queries = self.__class__.tokenizer.texts_to_sequences(records[:, 1])
            tokenized_queries = np.array([np.array(q, np.int64) for q in tokenized_queries])

            tokenized_queries_masked = self.__class__.tokenizer.texts_to_sequences(records[:, 2])
            tokenized_queries_masked = np.array([np.array(q, np.int64) for q in tokenized_queries_masked])

            tokenized_queries_masked_word = self.__class__.tokenizer.texts_to_sequences(records[:, 3])
            tokenized_queries_masked_word = np.array([np.array(q, np.int64) for q in tokenized_queries_masked_word])

            tokenized_queries_shuffled = self.__class__.tokenizer.texts_to_sequences(records[:, 4])
            tokenized_queries_shuffled = np.array([np.array(q, np.int64) for q in tokenized_queries_shuffled])

            return {
                'category': [cat2id[record] - 84 for record in records[:, 0]],
                # -82 because of targets shifted 82 to right side 82->0, 83->1 etc
                'query': tokenized_queries
            }


def convert_ints_to_words(ints):
    sentence_list = []
    sentences = []

    for sentence in ints:
        for i in sentence:
            i = i.item()
            if int(i) == 0:
                continue
            word = config.TOKENIZER.index_word[int(i)]
            if word == 1:
                continue
            sentence_list.append(config.TOKENIZER.index_word[int(i)])
        sentences.append(sentence_list)
        sentence_list = []

    for idx, s in enumerate(sentences):
        sentences[idx] = ' '.join(s)

    return sentences


def main():
    pathlib.Path('outputs/graphs/').mkdir(parents=True, exist_ok=True)

    # Choose supervised
    config.SUPERVISED = config.SUPERVISED_BANKING77

    # Choose phase 1
    config.PHASE_1 = config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS

    # Choose phase 2
    config.PHASE_2 = config.SEMI_SUPERVISED_PHASE_2_BANKING77

    # Choose supervised/phase 2 number of classes
    config.NUM_CLASSES = config.SUPERVISED_NUM_CLASSES_BANKING77

    words_datasets_used = [config.FILE_TRAINING,
                           config.FILE_VALIDATION,
                           config.FILE_TRAINING_BANKING77,
                           config.FILE_VALIDATION_BANKING77,
                           config.FILE_TRAINING_CLINC150,
                           config.FILE_VALIDATION_CLINC150]

    # Create WordsDatasets for training, validation and testing
    words_datasets = {x: WordsDataset(os.path.join(config.DATA_DIR, x))
                      for x in words_datasets_used}

    dataloaders = {x: DataLoader(words_datasets[x], batch_size=config.BATCH_SIZE,
                                 shuffle=True, num_workers=config.NUM_WORKERS,
                                 collate_fn=PadSequence())
                   for x in words_datasets_used}

    dataset_sizes = {x: len(words_datasets[x]) for x in dataloaders}

    # Run program on GPU if available, else run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config.DEVICE = device

    if config.FORCE_CPU:
        # Force running on CPU
        device = 'cpu'

    if device.type == 'cuda':
        print('[*] Using the GPU:', torch.cuda.get_device_name(device))
        if torch.cuda.device_count() > 1:
            print('[!] Multiple GPUs detected, only one device will be used')
    else:
        print('[!] Using the CPU')

    for model_num in config.MODEL_NUMS:
        #if model_num == 0 or model_num == 1:  # todo: remove later
            #continue

        # Supervised training/testing
        training_testing.run(device, dataset_sizes, dataloaders, config.NUM_CLASSES,
                             config.SUPERVISED, config.SUPERVISED_N_EPOCHS, model_num)

        # Semi-supervised phase 1 training/testing
        training_testing.run(device, dataset_sizes, dataloaders, config.VOCAB_SIZE,
                                                     config.PHASE_1, config.SEMI_SUPERVISED_PHASE_1_N_EPOCHS, model_num)

        #if model_semi_supervised is None:
            # SimpleModel and SimpleGRUModel does currently not support the "autoencoder" and "shuffled"
            # pre-processing techniques. For now, this is considered future work.
            #continue

        if model_num == 2:
            # Semi-supervised phase 2 training/testing
            training_testing.run(device, dataset_sizes, dataloaders, config.NUM_CLASSES,
                                 config.PHASE_2, config.SEMI_SUPERVISED_PHASE_2_N_EPOCHS,
                                 model_num)
        else:
            # Semi-supervised phase 2 training/testing
            training_testing.run(device, dataset_sizes, dataloaders, config.NUM_CLASSES,
                                 config.PHASE_2, config.SEMI_SUPERVISED_PHASE_2_N_EPOCHS,
                                 model_num)


def experiment_runner():

    # Choose supervised
    config.SUPERVISED = config.SUPERVISED_BANKING77

    # Choose phase 1
    config.PHASE_1 = config.SEMI_SUPERVISED_PHASE_1_MASKED_WORD_20NEWS

    # Choose phase 2
    config.PHASE_2 = config.SEMI_SUPERVISED_PHASE_2_BANKING77

    # Choose supervised/phase 2 number of classes
    config.NUM_CLASSES = config.SUPERVISED_NUM_CLASSES_BANKING77

    words_datasets_used = [config.FILE_TRAINING,
                           config.FILE_VALIDATION,
                           config.FILE_TRAINING_BANKING77,
                           config.FILE_VALIDATION_BANKING77,
                           config.FILE_TRAINING_CLINC150,
                           config.FILE_VALIDATION_CLINC150]

    # Get dataset size
    data_size = 0
    with utils.cwd(config.DATA_DIR):
        for file in (
                config.FILE_TRAINING,
                config.FILE_VALIDATION,
                config.FILE_TRAINING_BANKING77,
                config.FILE_VALIDATION_BANKING77,
                config.FILE_TRAINING_CLINC150,
                config.FILE_VALIDATION_CLINC150
        ):
            with open(file, encoding='utf-8') as f:
                data_size += sum(1 for line in f)

    random.seed(42)  # for predictability

    words_datasets = {x: WordsDataset(os.path.join(config.DATA_DIR, x))
                      for x in words_datasets_used}

    # Create Dataloaders for training, validation and testing
    dataloaders = {x: DataLoader(words_datasets[x], batch_size=config.BATCH_SIZE,
                                 shuffle=True, num_workers=config.NUM_WORKERS,
                                 collate_fn=PadSequence())
                   for x in words_datasets_used}

    dataset_sizes = {x: len(words_datasets[x]) for x in dataloaders}

    # 3 models
    models = range(3)
    # 1 preprocessing + 1 supervised
    preprocs = [config.SUPERVISED, config.PHASE_1]

    experiments = [{
        'size': data_size,
        'preproc': mutation[0],
        'model': mutation[1],
    } for mutation in product(preprocs, models)]

    # Make sure output folder exist
    pathlib.Path('outputs/experiments/').mkdir(parents=True, exist_ok=True)

    # Dismiss some experiments
    todo_experiments = list(dismiss_experiments(experiments))

    print(f'TODO: {len(todo_experiments)} experiments:')
    for experiment in todo_experiments:
        print(experiment)

    # Run program on GPU if available, else run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.FORCE_CPU:
        # Force running on CPU
        device = 'cpu'

    if device != 'cpu':
        print('[*] Using the GPU:', torch.cuda.get_device_name(device))
        if torch.cuda.device_count() > 1:
            print('[!] Multiple GPUs detected, only one device will be used')
    else:
        print('[!] Using the CPU')

    for experiment in todo_experiments:
        config._experiment = experiment  # hacky

        if experiment['preproc'] == config.SUPERVISED_BANKING77:
            config._experiment['phase1'] = False
            training_testing.run(device, dataset_sizes, dataloaders, config.NUM_CLASSES,
                                 experiment['preproc'], config.SUPERVISED_N_EPOCHS, experiment['model'])
            continue

        # Semi-supervised phase 1 training/testing
        config._experiment['phase1'] = True
        training_testing.run(device, dataset_sizes, dataloaders, config.VOCAB_SIZE,
                                                     experiment['preproc'],
                                                     config.SEMI_SUPERVISED_PHASE_1_N_EPOCHS, experiment['model'])

        #if model_semi_supervised is None:
            # SimpleModel and SimpleGRUModel does currently not support the "autoencoder" and "shuffled"
            # pre-processing techniques. For now, this is considered future work.
            #print('[!!!]')
            #continue

        config._experiment['phase1'] = False
        if experiment['model'] == 2:  # seq2seq
            # Semi-supervised phase 2 training/testing
            training_testing.run(device, dataset_sizes, dataloaders, config.NUM_CLASSES,
                                 config.PHASE_2, config.SEMI_SUPERVISED_PHASE_2_N_EPOCHS,
                                 experiment['model'])
        else:
            # Change last layer from classifying 11th word to categories
            #model_semi_supervised.fc = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES).to(device)

            # Semi-supervised phase 2 training/testing
            training_testing.run(device, dataset_sizes, dataloaders, config.NUM_CLASSES,
                                 config.PHASE_2, config.SEMI_SUPERVISED_PHASE_2_N_EPOCHS,
                                 experiment['model'])

    plot_viewer.save_plots()


def _experiment_to_filename_partial(experiment):
    preproc_map = {0: 'supervised', 1: 'eleventh', 2: 'shuffled', 3: 'masked', 4: 'autoenc',
                   9: 'masked_20news', 10: 'masked_clinc150'}
    model_map = {0: 'simple', 1: 'simplegru', 2: 'seq2seq'}
    return f'{preproc_map[experiment["preproc"]]}_{model_map[experiment["model"]]}'


def experiment_to_filename(experiment):
    return _experiment_to_filename_partial(experiment) + '.csv'


def experiment_to_filename_phase1(experiment):
    return _experiment_to_filename_partial(experiment) + '_preproc.csv'


def dismiss_experiments(experiments):
    for experiment in experiments:
        if not pathlib.Path(
                # Dismiss experiments we got the results from already
                'outputs/experiments/' + experiment_to_filename(experiment)
        ).is_file() and (
                # SimpleModel and SimpleGRUModel does currently not support the "autoencoder" and "shuffled"
                # pre-processing techniques. For now, this is considered future work.
                experiment['model'] not in (0, 1)
                or experiment['preproc'] not in (2, 4)
        ):
            yield experiment


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Run ML project.')
    parser.add_argument(
        '--experiments',
        action='store_const',
        const=True,
        default=False,
    )
    args = parser.parse_args()

    config._experiment = None  # hacky

    if args.experiments:
        experiment_runner()
    else:
        main()
