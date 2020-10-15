import argparse
import os

from config import SUPERVISED_CLASSES as valid_categories


def assert_continue(expr: bool, err_msg: str, filename: str, index: int):
    if not expr:
        print(f'[!] invalid {err_msg}, {filename}:{index+1}')


def verify_single(filename: str) -> int:
    counter = 0
    # Load dataset
    with open(filename, mode='r', encoding='utf-8') as f:
        # Verify the integrity of the dataset
        for line in f.readlines():
            cat, x, y = line.split(',')
            xs = x.split()

            # category must be valid
            assert_continue(
                cat in valid_categories,
                'category',
                filename,
                counter,
            )

            # input must be 10 words
            assert_continue(
                len(xs) == 10,
                'sentence',
                filename,
                counter,
            )

            # output must be 1 word
            assert_continue(
                ' ' not in y,
                '11th word',
                filename,
                counter,
            )
            # output must be a string
            assert_continue(
                type(y) is str,
                '11th word type',
                filename,
                counter,
            )

            counter += 1
    return counter


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
        counter = verify_single(filename)
        if args.debug:
            print(f'{filename} has {counter} datapoints')
