import os
from dataclasses import dataclass

from typing import List
import torch
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from attrdict import AttrDict
import collections


from sklearn.metrics import roc_curve, auc
import tqdm
import pandas as pd
import librosa
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score,
                                precision_score, f1_score,
                                roc_auc_score, confusion_matrix, classification_report)
import seaborn as sns


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




def plot_roc_curve(labels, logits):
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(15):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure()
    lw = 2
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'teal']
    for i, color in zip(range(15), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc[i])

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=lw, linestyle='--', label='Micro-average ROC curve (AUC = %0.2f)' % roc_auc["micro"])

    # Plot chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    # Set plot parameters
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_cnn_mfcc_test.png', dpi=300)
    plt.show()

def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor):
    """
    Mixup x of even indexes (0, 2, 4, ...)
    with x of odd indexes (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...), batch_size must be even.
      mixup_lambda: (batch_size * 2,).
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
    return out

class Mixup(object):
    def __init__(self, mixup_alpha):
        """
        Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha

    def get_lambda(self, batch_size):
        """
        Get mixup random coefficients.
        Args:
            batch_size: int
        Returns:
            mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return torch.from_numpy(np.array(mixup_lambdas, dtype=np.float32)).to(device)

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
    df_train, df_temp, y_train, y_temp = train_test_split(X, y,
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


def load_config(config_path: str) -> AttrDict:
    """Load config files and discard comments
    Args:
        config_path (str): path to config file.
    Source:
        https://github.com/Edresson/Wav2Vec-Wrapper
    """
    config = AttrDict()

    with open(config_path, "r", encoding="utf-8") as f:
        input_str = f.read()

    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)

    data = json.loads(input_str)
    config.update(data)

    return config

def plot_and_save_conf_matrix(data, especie_names, output_path):
    """Plots a confusion matrix based on a specific data
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set(font_scale=0.5)
    sns.set(rc={'figure.figsize':(16,8)})
    plot = sns.heatmap(data, annot=True, xticklabels=especie_names, yticklabels=especie_names, fmt='.3g', cmap="YlGnBu", center=np.mean(data), linewidths=.5)

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


def calc_dbfs_stats(
    df: pd.DataFrame,
    cfg: DictConfig
):
    dbfs_list = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        y, _ = librosa.load(
        os.path.join(cfg.data.data_path, row[cfg.metadata.audio_name_column_name]+".wav"),
        sr=cfg.train.sample_rate,
        mono=True,
        res_type='kaiser_best'
        )

        # Cut the audio in the right segment
        y = y[int(row[cfg.metadata.begin_time_column_name]*cfg.train.sample_rate) : int(row[cfg.metadata.end_time_column_name]*cfg.train.sample_rate)]

        rms = np.sqrt(np.mean(np.absolute(y)**2))
        dbfs = 20*np.log10(rms)

        dbfs_list.append(dbfs)

    print(f"DbFS Mean: {np.mean(dbfs_list)}")
