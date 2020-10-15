import os
import argparse
from config import SUPERVISED_CLASSES as valid_categories

def verify_single(filename, debug=True):
    counter = 0
    # Load dataset
    with open(filename, mode='r', encoding='utf-8') as f:
        # Verify the integrity of the dataset
        for line in f.readlines():
            cat, x, y = line.split(',')
            xs = x.split()

            # category must be valid
            assert cat in valid_categories

            # input must be 10 words
            assert len(xs) == 10

            # output must be 1 word
            assert ' ' not in y

            # output must be a string
            assert type(y) is str

            counter += 1

    if debug:
        print(f'{filename} has {counter} points')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Verify datasets.')
    parser.add_argument(
        'files',
        nargs='*',
        default=('words_test.csv', 'words_train.csv', 'words_val.csv'),
    )
    parser.add_argument(
        '--no-debug',
        dest='debug',
        action='store_const',
        const=False,
        default=True,
    )
    parser.add_argument(
        '--data-dir',
        default='data'
    )
    args = parser.parse_args()

    os.chdir(args.data_dir)
    for filename in args.files:
        verify_single(filename, debug=args.debug)