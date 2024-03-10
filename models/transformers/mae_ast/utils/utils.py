"""
    Mixup implementation based on:
        https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/750c318c0fcf089bd430f4d58e69451eec55f0a9/utils/utilities.py
        https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/tools/mixup.py
"""
from dataclasses import dataclass
from collections import namedtuple

import os
import tqdm
import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np
from utils.data_generator import DataGenerator


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class Colors(metaclass=Singleton):
    BLACK: str = '\033[30m'
    RED: str = '\033[31m'
    GREEN: str = '\033[32m'
    YELLOW: str = '\033[33m'
    BLUE: str = '\033[34m'
    MAGENTA: str = '\033[35m'
    CYAN: str = '\033[36m'
    WHITE: str = '\033[37m'
    UNDERLINE: str = '\033[4m'
    RESET: str = '\033[0m'


@dataclass(frozen=True)
class LogFormatter(metaclass=Singleton):
    colors_single = Colors()
    TIME_DATA: str = colors_single.BLUE + '%(asctime)s' + colors_single.RESET
    MODULE_NAME: str = colors_single.CYAN + '%(module)s' + colors_single.RESET
    LEVEL_NAME: str = colors_single.GREEN + '%(levelname)s' + colors_single.RESET
    MESSAGE: str = colors_single.WHITE + '%(message)s' + colors_single.RESET
    FORMATTER = '['+TIME_DATA+']'+'['+MODULE_NAME+']'+'['+LEVEL_NAME+']'+' - '+MESSAGE


formatter_single = LogFormatter()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
Info = namedtuple("Info", ["length", "sample_rate", "channels"])

def calculate_stats_norm(dataset, cfg):

    data_gen = DataGenerator(
        batch=dataset,
        sample_rate=cfg.feature_extractor.sample_rate,
        period=cfg.data.period,
        type='val',
        freqm=None,
        timem=None,
        skip_norm=True,
        norm_mean=None,
        norm_std=None,
        use_specaug=False,
        target_length=cfg.feature_extractor.target_length,
        rand_sampling=False,
        use_audio_mixup=False,
        mixup_alpha=None,
        insert_noise=False,
        class_num=cfg.train.classes_num
    )

    data_loader = torch.utils.data.DataLoader(
        data_gen,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    mean=[]
    std=[]

    for sample in tqdm.tqdm(data_loader):
        cur_mean = torch.mean(sample['image'])
        cur_std = torch.std(sample['image'])
        mean.append(cur_mean)
        std.append(cur_std)

    return np.mean(mean), np.mean(std)

def get_audio_info(path: str):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)

# Receives the predictions and true labels of a multiclass classifier and plot a reability diagram
def reliability_diagram(y_true, y_pred, n_bins=10, title='Reliability Diagram', save_path=None):
    """
    Plot a reliability diagram
    :param y_true: true labels
    :param y_pred: predicted labels
    :param n_bins: number of bins
    :param title: plot title
    :param save_path: path to save the plot
    :return:
    """

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get the confidence of the predictions
    confidences = np.max(y_pred, axis=1)

    # Get the predicted labels
    y_pred = np.argmax(y_pred, axis=1)

    # print(y_pred, type(y_pred))

    # # Get the true labels
    # y_true = np.argmax(y_true, axis=1)

    y_true = np.array(y_true)

    # Get the bin edges
    bin_edges = np.linspace(0., 1. + 1e-8, n_bins + 1)

    # Get the bin centers
    bin_centers = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)])

    # Get the bin indices for each prediction
    bin_indices = np.digitize(confidences, bin_edges[:-1])

    # Get the number of predictions in each bin
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(1, len(bin_edges))])

    # Get the average confidence in each bin
    bin_confidences = np.array([np.mean(confidences[bin_indices == i]) for i in range(1, len(bin_edges))])

    # Get the accuracy in each bin
    bin_accuracies = np.array([np.mean(y_pred[bin_indices == i] == y_true[bin_indices == i]) for i in range(1, len(bin_edges))])

    # Plot the reliability diagram
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.plot(bin_confidences, bin_accuracies, 's-', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()