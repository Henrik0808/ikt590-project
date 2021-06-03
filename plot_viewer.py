import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

render_opts = {'save': True, 'show': False}
paths_supervised = []
paths_preproc = []
paths_downstream = []
paths_supervised_downstream = [[], [], []]


def render_plot_preproc(preproc):
    print(preproc)
    epoch, loss_train, loss_valid = np.genfromtxt(
        preproc, delimiter=',', skip_header=True, unpack=True)

    fig, ax = plt.subplots()
    ax.set_title(preproc.stem)
    ax.plot(epoch, loss_train, label='Training')
    ax.plot(epoch, loss_valid, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend()

    if render_opts['show']:
        plt.show()
    if render_opts['save']:
        preproc = preproc.with_suffix('.png')
        file_name = preproc.parts[0] + '/' + 'graphs/' + preproc.name
        plt.savefig(file_name, dpi=300)


def render_plot_sup_down(sup, down):
    print(down, sup)
    epoch_sup, loss_train_sup, loss_valid_sup = np.genfromtxt(
        sup, delimiter=',', skip_header=True, unpack=True)

    epoch_down, loss_train_down, loss_valid_down = np.genfromtxt(
        down, delimiter=',', skip_header=True, unpack=True)

    fig, ax = plt.subplots()
    ax.set_title(down.stem)
    ax.plot(epoch_sup, loss_train_sup, label='Supervised training')
    ax.plot(epoch_down, loss_train_down, label='Downstream training')
    ax.plot(epoch_sup, loss_valid_sup, label='Supervised validation')
    ax.plot(epoch_down, loss_valid_down, label='Downstream validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend()

    if render_opts['show']:
        plt.show()
    if render_opts['save']:
        down = down.with_suffix('.png')
        file_name = down.parts[0] + '/' + 'graphs/' + down.name
        plt.savefig(file_name, dpi=300)


def parse_paths(paths):
    for path in paths:
        if path.is_file():
            if 'supervised' in path.name:
                paths_supervised.append(path)

                if 'simple.' in path.name:
                    paths_supervised_downstream[0].append(path)
                elif 'simplegru' in path.name:
                    paths_supervised_downstream[1].append(path)
                else:
                    paths_supervised_downstream[2].append(path)
            elif 'preproc' in path.name:
                paths_preproc.append(path)
            else:
                paths_downstream.append(path)

                if 'simple.' in path.name:
                    paths_supervised_downstream[0].append(path)
                elif 'simplegru' in path.name:
                    paths_supervised_downstream[1].append(path)
                else:
                    paths_supervised_downstream[2].append(path)
        elif path.is_dir():
            parse_paths(path.iterdir())
        else:
            print(f'Path @ "{path}" not found.', file=sys.stderr)


def save_plots():
    parse_paths(pathlib.Path(path) for path in [pathlib.Path('outputs/experiments/')])

    for sup_down in paths_supervised_downstream:
        if not sup_down:
            continue

        down = sup_down[0]
        sup = sup_down[1]

        render_plot_sup_down(sup, down)

    for preproc in paths_preproc:
        render_plot_preproc(preproc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot CSV files',
                                     description='example: python plot_viewer.py --save --no-show')
    parser.add_argument('--save', action='store_true', help='save figure as an image', default=True)
    parser.add_argument('--no-show', action='store_false', dest='show', help='do not display the plot', default=False)
    args = parser.parse_args()

    render_opts = {'save': args.save, 'show': args.show}

    save_plots()
