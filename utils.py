import os
import random
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt


def process_data(data, data_slice, iter):
    warnings.filterwarnings('error')
    for key in data.keys():
        if key == 'RA':
            if iter == 0:  # Save first iteration info only.
                data[key] = data_slice[key]
        elif type(data_slice[key]) is list:
            try:
                data[key] = iter * data[key] / (iter + 1) + np.mean(data_slice[key]) / (iter + 1)  # Average.
            except RuntimeWarning:
                data[key] = iter * data[key] / (iter + 1)  # Average.
        else:
            data[key] = iter * data[key] / (iter + 1) + data_slice[key] / (iter + 1)  # Average.

    return data


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Counter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0  # number of total values of the counter
        self.true = 0  # number of true values of the counter

    def update(self, tf):
        if len(tf.shape) == 0:
            self.count += 1
        else:
            self.count += len(tf)
        self.true += torch.sum(tf).item()

    def get(self):
        return self.true / self.count


def plot_agent_data(agent):
    title_list = ['$\mu$', '$\\nu$', 'Primal & Dual function', 'Primal - Dual function']
    legend_list = ['$\mu$', '$\\nu$', 'Primal', 'Dual', 'Primal - Dual']
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    axes = axes.flatten()

    # Plot mu and nu
    axes[0].plot(agent.record[:, 0], label=legend_list[0])
    axes[1].plot(agent.record[:, 1], label=legend_list[1])

    # Plot primal and dual
    axes[2].plot(agent.record[:, 2], label=legend_list[2])
    axes[2].plot(agent.record[:, 3], label=legend_list[3])

    # Plot primal - dual
    axes[3].plot(agent.record[:, 2] - agent.record[:, 3], label=legend_list[4])

    for i in range(4):
        axes[i].set_title(f"{title_list[i]}")
        axes[i].set_xlabel('Percentage complete')
        axes[i].legend()
    plt.show()

    print("Plotting finished.")


def plot_lr_data(data_list, save_path, num_trials, vars_list, header, save=True):
    title_list = ['$\mu$', '$\\nu$', 'Primal & Dual function', 'Primal - Dual function']
    legend_list = ['$\mu$', '$\\nu$', 'Primal', 'Dual', 'Primal - Dual']
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    axes = axes.flatten()

    fig.suptitle(f"Learning rate comparison ({num_trials} trials)")

    for i, data in enumerate(data_list):
        # Plot mu and nu
        axes[0].plot(data[:, 0], label=f"{vars_list[i]}")
        axes[1].plot(data[:, 1], label=f"{vars_list[i]}")

        # Plot primal and dual
        axes[2].plot(data[:, 2], label=f"Primal ({vars_list[i]})")
        axes[2].plot(data[:, 3], label=f"Dual ({vars_list[i]})")

        # Plot primal - dual
        axes[3].plot(data[:, 2] - data[:, 3], label=f"{legend_list[4]} ({vars_list[i]})")

    for i in range(4):
        axes[i].set_title(f"{title_list[i]}")
        axes[i].set_xlabel('Percentage complete')
        axes[i].legend()
    if save:
        plt.savefig(os.path.join(save_path, f"{header}.png"))
    else:
        plt.show()

    pass
