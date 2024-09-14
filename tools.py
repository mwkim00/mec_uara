import random
import warnings

import numpy as np
import torch


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


def plot_nu(agent):
    import matplotlib.pyplot as plt

    for i in range(agent.record.shape[1]):
        for j in range(agent.record.shape[2]):
            plt.figure()
            plt.plot(agent.record[:, i, j])
            plt.title(f"nu idx: {i+1} | BS idx: {j+1}")
            plt.show()
