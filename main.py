import os
import pathlib

import pandas as pd
import torch
import torch.nn as nn
from keras_preprocessing import sequence, text
from torch.utils.data import DataLoader, Dataset

import config
import training_testing


class WordsDataset(Dataset):
    """Words dataset containing training, validation or testing data."""

    # The tokenizer is created from the 10-word sentences in the training dataset.
    # The same tokenizer is used in all three dataset instances, to use the same indexes/tokens for the same words etc.
    tokenizer = None

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory containing all three raw words datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample. Probably not needed here.
        """
        self.ten_words_tokenized, self.categories_ints, self.eleventh_word = self.get_datasets(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ten_words_tokenized)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        category = self.categories_ints[idx]
        ten_words = self.ten_words_tokenized[idx]
        eleventh_word = self.eleventh_word[idx]

        sample = {'10 words': ten_words, 'category': category, 'eleventh word': eleventh_word}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_datasets(self, csv_file):
        words = None
        ten_word_sentences_list_of_strings = None

        # if tokenizer is None, create it. Else use the already created tokenizer
        if self.tokenizer is None:
            words, ten_word_sentences_list_of_strings = self.create_train_tokenizer(csv_file)

        # Load csv rows into the words variable, if it's None
        if words is None:
            words = pd.read_csv(csv_file)  # If you want to load fewer rows: add parameter nrows=1000 for example

        ten_word_sentences_list_of_lists_with_category = words.iloc[:, [0]].values
        ten_word_sentences_list_of_lists_with_eleventh_word = words.iloc[:, [2]].values

        # Load the 10-word sentences into the ten_word_sentences_list_of_strings variable, if not already loaded
        if ten_word_sentences_list_of_strings is None:
            ten_word_sentences_list_of_lists_with_string = words.iloc[:, [1]].values
            ten_word_sentences_list_of_strings = [s[0] for s in ten_word_sentences_list_of_lists_with_string]

        ten_word_sentences_list_of_categories = [s[0] for s in ten_word_sentences_list_of_lists_with_category]

        # 0: sports, 1: religion, 2: computers
        ten_word_sentences_list_of_categories_as_ints = [0 if c == 'sports' else (1 if c == 'religion' else 2) for c in
                                                         ten_word_sentences_list_of_categories]

        list_of_eleventh_word = [s[0] for s in ten_word_sentences_list_of_lists_with_eleventh_word]

        # Make sure list of eleventh words contains only strings
        # TODO: Fix bug causing eleventh word to not always be a string
        list_of_eleventh_word = [w if isinstance(w, str) else str(w) for w in list_of_eleventh_word]

        eleventh_word_tokenized = self.tokenizer.texts_to_sequences(list_of_eleventh_word)

        for i in eleventh_word_tokenized:
            # If tokenized eleventh word list is empty, append 1 for now
            # TODO: Tokenizer removes/ignores certain 'words', such as '-'. This causes some i's here to be empty
            if not i:
                i.append(1)  # 1 is used by tokenizer for appending unknown word/index

        eleventh_word_tokenized = [i[0] for i in eleventh_word_tokenized]

        ten_word_sentences_tokenized = self.tokenizer.texts_to_sequences(ten_word_sentences_list_of_strings)

        # By default, index 0 is used as padding token on the left side of the sentence
        # TODO: Fix bug with 'ten word sentence' sometimes not having length 10
        ten_word_sentences_tokenized = sequence.pad_sequences(ten_word_sentences_tokenized,
                                                              maxlen=10)

        return ten_word_sentences_tokenized, ten_word_sentences_list_of_categories_as_ints, eleventh_word_tokenized

    @staticmethod
    def create_train_tokenizer(csv_file):

        # Create tokenizer
        words = pd.read_csv(csv_file)

        # Unknown words will have index 1
        tokenizer = text.Tokenizer(oov_token=1)

        ten_word_sentences_list_of_lists_with_string = words.iloc[:, [1]].values
        ten_word_sentences_list_of_strings = [s[0] for s in ten_word_sentences_list_of_lists_with_string]

        tokenizer.fit_on_texts(ten_word_sentences_list_of_strings)

        config.VOCAB_SIZE = len(tokenizer.word_index) + 1  # TODO: remove +1 if pad_sequences is not used

        # Update the tokenizer class variable
        WordsDataset.tokenizer = tokenizer

        return words, ten_word_sentences_list_of_strings


def main():
    pathlib.Path('outputs/graphs/supervised_training').mkdir(parents=True, exist_ok=True)
    pathlib.Path('outputs/graphs/semi_supervised_training/phase_one').mkdir(parents=True, exist_ok=True)
    pathlib.Path('outputs/graphs/semi_supervised_training/phase_two').mkdir(parents=True, exist_ok=True)

    # Create WordsDatasets for training, validation and testing
    words_datasets = {x: WordsDataset(os.path.join(config.DATA_DIR, x), config.DATA_DIR)
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
    model_semi_supervised.fc = nn.Linear(config.HIDDEN_DIM, config.SEMI_SUPERVISED_PHASE_2_NUM_CLASSES).to(device)

    # Semi-supervised phase 2 training/testing
    training_testing.run(device, dataset_sizes, dataloaders, config.SUPERVISED_NUM_CLASSES,
                         config.SEMI_SUPERVISED_PHASE_2, config.SEMI_SUPERVISED_PHASE_2_N_EPOCHS,
                         model_semi_supervised)


if __name__ == '__main__':
    main()
