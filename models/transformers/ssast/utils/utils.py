"""
    Mixup implementation based on:
        https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/750c318c0fcf089bd430f4d58e69451eec55f0a9/utils/utilities.py
        https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/tools/mixup.py
"""
from dataclasses import dataclass
from collections import namedtuple

import tqdm
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