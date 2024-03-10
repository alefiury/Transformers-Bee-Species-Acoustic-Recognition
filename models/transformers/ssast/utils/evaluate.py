import os
import logging
from functools import reduce
from operator import concat

import pandas as pd
import torch
import numpy as np
from sklearn import metrics
import timm

from tqdm import tqdm
import librosa

# from model import Cnn14, Transfer_Cnn14

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ast_models import ASTModel

from utils.utils import formatter_single

device = ('cuda' if torch.cuda.is_available() else 'cpu')

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)

def test_model(df_test, checkpoint_path, cfg):
    """
    Predicts new data.
    """

    print(checkpoint_path)
    print(os.path.isfile(checkpoint_path))

    model = ASTModel(
        label_dim=cfg.train.classes_num,
        fshape=cfg.model.fshape,
        tshape=cfg.model.tshape,
        fstride=cfg.model.fstride,
        tstride=cfg.model.tstride,
        input_fdim=cfg.feature_extractor.mel_bins,
        input_tdim=cfg.feature_extractor.target_length,
        model_size=cfg.model.model_size,
        pretrain_stage=False,
        load_pretrained_mdl_path=cfg.model.pretrained_checkpoint_path
    )

    model.to(device)

    pred_list = []
    labels = []
    diff = []

    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    with torch.no_grad():
        model.eval()

        for test_sample in tqdm(df_test):
            test_audio, test_label = test_sample['image'].to(device), test_sample['hot_target'].to(device)
            output = model(
                test_audio,
                cfg.train.task
            )
            output = torch.sigmoid(output)
            out = torch.argmax(output, dim=1).cpu().detach().numpy().tolist()
            label = torch.argmax(test_label, dim=1).cpu().detach().numpy().tolist()

            pred_list.append(out)
            labels.append(label)

    pred_list  = reduce(concat, pred_list)
    labels  = reduce(concat, labels)

    # for p, l in zip(pred_list, labels):
    #     if p!=l:
    #         diff.append([p, l])

    # print(diff)


    # f1_score = metrics.f1_score(np.array(labels), np.array(pred_list), average='macro')
    # precision = metrics.precision_score(np.array(labels), np.array(pred_list), average='macro')
    # recall = metrics.recall_score(np.array(labels), np.array(pred_list), average='macro')
    # acc = metrics.accuracy_score(np.array(labels), np.array(pred_list))

    # print(f"Acc: {acc} | recall: {recall} | precision: {precision} | f1-Score: {f1_score}")

    # print(metrics.classification_report(np.array(labels), np.array(pred_list), target_names=languages))

    # cm_analysis(y_true=np.array(labels), y_pred=np.array(pred_list), filename=os.path.join(original_cwd, 'confusion_matrix.png'), labels=[0, 1, 2, 3, 4, 5], ymap=None, figsize=(10,10))

    # confusion_matrix = metrics.confusion_matrix(np.array(labels), np.array(pred_list), labels=[0, 1, 2, 3, 4, 5])
    # display = metrics.ConfusionMatrixDisplay(confusion_matrix).plot()
    # display.savefig(os.path.join(original_cwd, 'confusion_matrix.png'))

    # return acc, recall, precision, f1_score

    return labels, pred_list