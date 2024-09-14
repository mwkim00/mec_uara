import argparse
import os
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_col(file_name, col='Average Delay', save_fig=False):
    label_fontsize = 15
    title_fontsize = 15

    file_name = os.path.join('log', file_name)
    df = pd.read_csv(file_name)
    plt.figure(figsize=(8, 6))

    sns.lineplot(x='Number of Users', y=col, hue='Scheme', data=df, palette="tab10",
                 linewidth=1)
    plt.xlabel("Number of Users", fontsize=label_fontsize)
    plt.ylabel("Average Delay (ms)", fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    if save_fig:
        os.makedirs('./figs', exist_ok=True)
        plt.savefig(os.path.join('figs', file_name[:-4] + '.png'))
    else:
        plt.show()


def plot_multi_file(files=None, x=1, y=2, save_fig=False):
    """
    x, y could be either the column index, or the name of the column. Refer to the list below for column info.
    labels = ['Scheme', 'Number of Users', 'Average Delay', 'Local 1', 'Local 2', 'Edge']
    """
    if type(files) is str:
        file_names = ['Random.csv', 'MaxSINR.csv', files]
    elif type(files) is list:
        file_names = ['Random.csv', 'MaxSINR.csv'] + files
    else:
        file_names = ['Random.csv', 'MaxSINR.csv', '1e10.csv']

    label_fontsize = 15
    title_fontsize = 15
    log_dir = os.path.join(os.getcwd(), 'log')
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=label_fontsize)

    file_list = [pd.read_csv(os.path.join(log_dir, file_names[i])) for i in range(len(file_names))]
    if type(x) is int:
        for (i, file) in enumerate(file_list):
            label = file.iloc[:, 0][0]
            if label == 'Proposed':
                label = f"Proposed (mu = {file_names[i][:-4]})"
            plt.plot(file.iloc[:, x], file.iloc[:, y], label=label)

        plt.xlabel(file.iloc[:, x].name, fontsize=label_fontsize)
        plt.ylabel(file.iloc[:, y].name, fontsize=label_fontsize)
    else:
        for (i, file) in enumerate(file_list):
            plt.plot(file[x], file[y], label=file.iloc[:, 0][0])

        plt.xlabel(x, fontsize=label_fontsize)
        plt.ylabel(y, fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    if save_fig:
        os.makedirs('./figs', exist_ok=True)
        plt.savefig(os.path.join('figs', datetime.now().strftime("%Y%m%d%H%M") + '.png'), dpi=200)
    else:
        plt.show()


def plot_average_delay_fix(files=None, save_fig=False):
    """
    A fix of plot_multi_file(), with x='Number of Users', y='Average Delay'.
    Ignore delays that have been caused by packets exclusively made for local. DEBUG: is it necessary
    """
    label_fontsize = 15
    title_fontsize = 15
    log_dir = os.path.join(os.getcwd(), 'log')
    plt.figure(figsize=(8, 6))
    if type(files) is str:
        file_list = ['RandAgent.csv', 'MaxSINRAgent.csv', files]
    elif type(files) is list:
        file_list = files
    else:
        file_list = ['RandAgent.csv', 'MaxSINRAgent.csv', 'Proposed_1.9e08.csv']
    file_list = [pd.read_csv(os.path.join(log_dir, file_list[i])) for i in range(len(file_list))]

    for (i, file) in enumerate(file_list):
        avg_delay = file['Average Delay']
        avg_delay = avg_delay * (file['Loc 1'] + file['Loc 2'] + file['Loc 3']) - 0.003 * 1000 * file['Loc 1']
        avg_delay = avg_delay / (file['Loc 2'] + file['Loc 3'])
        plt.plot(file['Number of Users'], avg_delay, label=file.iloc[:, 0][0])
        plt.xlabel('Number of Users', fontsize=label_fontsize)
        plt.ylabel('Average Delay', fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    if save_fig:
        os.makedirs('./figs', exist_ok=True)
        plt.savefig(os.path.join('figs', file_name[:-4] + '.png'), dpi=300)
    else:
        plt.show()


"""
To save figure, use --save-fig or -s.
Else, use --no-save-fig or --no-s.
"""
parser = argparse.ArgumentParser(description='Visualize log')
parser.add_argument('--file-name', '-f', type=str, default='Random.csv', help='File name of the log')
parser.add_argument('--version', '-v', type=int, default=0, help='Plotting function version')
parser.add_argument('--save-fig', '-s', action=argparse.BooleanOptionalAction, default=False, help='Save figure')
args = parser.parse_args()


def main(file_name=None, version=None, save_fig=None):
    funcs = [plot_single_col, plot_multi_file, plot_average_delay_fix]

    if file_name is None:
        file_name = args.file_name
    if version is None:
        version = args.version
    if save_fig is None:
        save_fig = args.save_fig

    # Change settings by hand
    file_name = None
    version = 1
    save_fig = save_fig

    funcs[version](file_name, save_fig=save_fig)

    # plot_multi_file(files=['RandAgent.csv', 'MaxSINRAgent.csv'], x='Number of Users', y='Average Delay')
    # plot_average_delay(files=['RandAgent.csv', 'MaxSINRAgent.csv', 'Proposed_1.9e08.csv'])
    # plot_average_delay(files=['Proposed_1.1e08.csv', 'Proposed_1.5e08.csv', 'Proposed_1.9e08.csv'])


if __name__ == '__main__':
    main()
