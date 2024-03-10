import os
import glob
import logging
import argparse

import torch
import wandb
import numpy as np
import pandas as pd
from sklearn import metrics
from omegaconf import OmegaConf
from datasets import load_from_disk

from utils.utils import formatter_single, calculate_stats_norm
from utils.data_generator import DataGenerator
from utils.evaluate import test_model
from trainer import train_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='config/default.yaml',
        type=str,
        help="YAML file with configurations"
    )

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    seeds = cfg.data.seeds

    mean_norms = []
    std_norms = []

    for idx, seed in enumerate(seeds):
        for cv in ['cv1', 'cv2']:
            log.info(f'Seed: {seed} | CV: {cv} -> {idx+1}/{len(cfg.data.seeds)}')

            train_path = os.path.join(cfg.data.preloaded_data_path, f'seed_{seed}', cv, 'train')
            val_path = os.path.join(cfg.data.preloaded_data_path, f'seed_{seed}', cv, 'val')

            preloaded_train_dataset = load_from_disk(train_path)
            preloaded_val_dataset = load_from_disk(val_path)

            preloaded_train_dataset.set_format(type='torch', columns=['image', 'target'])
            preloaded_val_dataset.set_format(type='torch', columns=['image', 'target'])

            train_mean_norm, train_std_norm = calculate_stats_norm(preloaded_train_dataset, cfg)

            mean_norms.append(train_mean_norm)
            std_norms.append(train_std_norm)

    print(np.mean(mean_norms), np.mean(std_norms))

if __name__ == '__main__':
    main()