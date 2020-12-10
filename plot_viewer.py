import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

render_opts = {}


def render_plot(file):
    print(file)
    epoch, loss_train, loss_valid = np.genfromtxt(
        file, delimiter=',', skip_header=True, unpack=True)

    fig, ax = plt.subplots()
    ax.set_title(file.stem)
    ax.plot(epoch, loss_train, label='Training')
    ax.plot(epoch, loss_valid, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend()

    if render_opts['show']:
        plt.show()
    if render_opts['save']:
        plt.savefig(file.with_suffix('.png'), dpi=300)


def parse_paths(paths):
    for path in paths:
        if path.is_file():
            render_plot(path)
        elif path.is_dir():
            parse_paths(path.iterdir())
        else:
            print(f'Path @ "{path}" not found.', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot CSV files',
        description='example: python plot_viewer.py --save --no-show outputs/experiments/')
    parser.add_argument('--save', action='store_true', help='save figure as an image')
    parser.add_argument('--no-show', action='store_false', dest='show', help='do not display the plot')
    parser.add_argument('path', nargs='*')
    args = parser.parse_args()

    render_opts = {'save': args.save, 'show': args.show}
    parse_paths(pathlib.Path(path) for path in args.path)
