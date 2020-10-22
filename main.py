import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
from keras_preprocessing import text
from torch.utils.data import DataLoader, Dataset

import config
import training_testing
import utils


class WordsDataset(Dataset):
    """Words dataset containing training, validation or testing data."""

    # The tokenizer is created from the 10-word sentences in the training dataset.
    # The same tokenizer is used in all three dataset instances, to use the same indexes/tokens for the same words etc.
    tokenizer = None

    @staticmethod
    def create_tokenizer(records):
        # Unknown words will have index 1
        tokenizer = text.Tokenizer(oov_token=1, filters='')
        # Join 10 word sentence and 11th word by space
        tokenizer.fit_on_texts(' '.join(record) for record in records[:, 1:])
        # Adding 1 to vocabulary size because of additional reserved padding index 0
        config.VOCAB_SIZE = len(tokenizer.word_index) + 1
        config.SOS_TOKEN = tokenizer.word_index['sos']
        return tokenizer

    @staticmethod
    def read_records(dataset_file):
        with open(dataset_file, mode='r', encoding='utf-8') as f:
            return np.array([
                line.split(',') for line in
                (line.rstrip() for line in f.readlines())
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
            self.__class__.tokenizer =\
                self.__class__.create_tokenizer(records)

        self.records_tokenized = self.tokenize(records)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):  # ??? -- is this in use?
            idx = idx.tolist()

        sample = {k: v[idx] for k, v in self.records_tokenized.items()}

        if self.transform:
            return self.transform(sample)

        sample['10 words'] = sample['10 words'][1:]

        return sample

    def tokenize(self, records):
        _, cat2id, _ = utils.get_categories()

        tokenized = np.asarray(self.__class__.tokenizer.texts_to_sequences(
            ' '.join(record) for record in records[:, 1:]
        ), dtype=np.int64)

        return {
            'category': [cat2id[record] for record in records[:, 0]],
            '10 words': tokenized[:, :-1],
            'eleventh word': tokenized[:, -1],
        }


def main():
    pathlib.Path('outputs/graphs/supervised_training').mkdir(parents=True, exist_ok=True)
    pathlib.Path('outputs/graphs/semi_supervised_training/phase_one').mkdir(parents=True, exist_ok=True)
    pathlib.Path('outputs/graphs/semi_supervised_training/phase_two').mkdir(parents=True, exist_ok=True)

    # Create WordsDatasets for training, validation and testing
    words_datasets = {x: WordsDataset(os.path.join(config.DATA_DIR, x))
                      for x in [config.FILE_TRAINING, config.FILE_VALIDATION, config.FILE_TESTING]}

    # Create Dataloaders for training, validation and testing
    dataloaders = {x: DataLoader(words_datasets[x], batch_size=config.BATCH_SIZE,
                                 shuffle=True, num_workers=config.NUM_WORKERS)
                   for x in [config.FILE_TRAINING, config.FILE_VALIDATION, config.FILE_TESTING]}

    dataset_sizes = {x: len(words_datasets[x]) for x in [config.FILE_TRAINING,
                                                         config.FILE_VALIDATION,
                                                         config.FILE_TESTING]}

    # Run program on GPU if available, else run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Uncomment the line below to force running on CPU
    # device = 'cpu'

    if device.type == 'cuda':
        print('[*] Using the GPU:', torch.cuda.get_device_name(device))
        if torch.cuda.device_count() > 1:
            print('[!] Multiple GPUs detected, only one device will be used')
    else:
        print('[!] Using the CPU')

    # Supervised training/testing
    training_testing.run(device, dataset_sizes, dataloaders, config.SUPERVISED_NUM_CLASSES,
                         config.SUPERVISED, config.SUPERVISED_N_EPOCHS)

    # Semi-supervised phase 1 training/testing
    model_semi_supervised = training_testing.run(device, dataset_sizes, dataloaders, config.VOCAB_SIZE,
                                                 config.SEMI_SUPERVISED_PHASE_1,
                                                 config.SEMI_SUPERVISED_PHASE_1_N_EPOCHS)

    # Change last layer from classifying 11th word to categories
    model_semi_supervised.decoder.fc = nn.Linear(config.HIDDEN_DIM,
                                                 config.SEMI_SUPERVISED_PHASE_2_NUM_CLASSES).to(device)

    # Semi-supervised phase 2 training/testing
    training_testing.run(device, dataset_sizes, dataloaders, config.SUPERVISED_NUM_CLASSES,
                         config.SEMI_SUPERVISED_PHASE_2, config.SEMI_SUPERVISED_PHASE_2_N_EPOCHS,
                         model_semi_supervised)


if __name__ == '__main__':
    main()
