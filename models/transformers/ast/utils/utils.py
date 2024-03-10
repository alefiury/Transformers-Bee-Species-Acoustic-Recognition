import os
import re
from collections import namedtuple
from typing import List

import torch
import tqdm
import torchaudio
import json
import numpy as np
import matplotlib.pyplot as plt
from attrdict import AttrDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score,
                                precision_score, f1_score,
                                roc_auc_score, confusion_matrix)
import seaborn as sns

from .data_generator import DataGenerator

device = ('cuda' if torch.cuda.is_available() else 'cpu')

Info = namedtuple("Info", ["length", "sample_rate", "channels"])

def split_stratified_into_train_val_test(df_input, stratify_colname='especie_id',
                                         frac_train=0.80, frac_val=0.10, frac_test=0.10,
                                         random_state=None):
    '''
    source: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def plot_and_save_conf_matrix(data, especie_names, output_path):
    """Plots a confusion matrix based on a specific data
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set(font_scale=0.5)
    sns.set(rc={'figure.figsize':(12,8)})
    plot = sns.heatmap(data, annot=True, xticklabels=especie_names, yticklabels=especie_names, fmt='.3g')

    figure1 = plot.get_figure()
    plt.tight_layout()
    figure1.savefig(output_path)

    plt.close()

def calculate_test_metrics(preds, labels, especie_names, model_type):
    accs = []
    recalls = []
    precisions = []
    f1_scores = []
    conf_ms = []

    for pred, label in zip(preds, labels):
        accs.append(accuracy_score(label, pred))
        recalls.append(recall_score(label, pred, average='macro'))
        precisions.append(precision_score(label, pred, average='macro'))
        f1_scores.append(f1_score(label, pred, average='macro'))
        conf_ms.append(confusion_matrix(label, pred))

    print(f"Acc mean: {np.mean(accs)*100} | Acc std: {np.std(accs)*100} | Acc min: {np.min(accs)*100} | Acc max: {np.max(accs)*100}")
    print(f"Recall mean: {np.mean(recalls)*100} | Recall std: {np.std(recalls)*100} | Recall min: {np.min(recalls)*100} | Recall max: {np.max(recalls)*100}")
    print(f"Precision mean: {np.mean(precisions)*100} | Precision std: {np.std(precisions)*100} | Precision min: {np.min(precisions)*100} | Precision max: {np.max(precisions)*100}")
    print(f"F1-Score mean: {np.mean(f1_scores)*100} | F1-Score std: {np.std(f1_scores)*100} | F1-Score min: {np.min(f1_scores)*100} | F1-Score max: {np.max(f1_scores)*100}")

    plot_and_save_conf_matrix(data=np.mean(conf_ms, axis=0),
                                especie_names=especie_names,
                                output_path=os.path.join('confusion_matrix',
                                    model_type, 'confusion_matrix.png'))

    plot_and_save_conf_matrix(data=np.std(conf_ms, axis=0),
                                especie_names=especie_names,
                                output_path=os.path.join('confusion_matrix',
                                    model_type, 'confusion_matrix_std.png'))

def calculate_stats_norm(dataset, sample_rate, period, batch_size, num_workers):

    data_gen = DataGenerator(dataset,
                                sample_rate,
                                period,
                                type='val',
                                skip_norm=True)

    data_loader = torch.utils.data.DataLoader(data_gen,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=num_workers)

    mean=[]
    std=[]

    for sample in tqdm.tqdm(data_loader):
        cur_mean = torch.mean(sample['image'])
        cur_std = torch.std(sample['image'])
        mean.append(cur_mean)
        std.append(cur_std)

    return np.mean(mean), np.mean(std)

def get_audio_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)