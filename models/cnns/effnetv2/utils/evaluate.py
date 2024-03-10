import functools
import itertools
import operator

import timm
import torch
import librosa
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from models.cnns.effnetv2.model import Cnn_Model

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# def make_batch_test_data(sample, period):
#     batch = []
#     sample = sample.astype(np.float32)
#     len_y = len(sample)
#     # print(len_y//32000)
#     start = 0
#     effective_lenght = period * config.config.sample_rate
#     end = effective_lenght
#     while True:
#         y_batch = sample[start:end].astype(np.float32)
#         if len(y_batch) != effective_lenght:
#             y_pad = np.zeros(effective_lenght, dtype=np.float32)
#             y_pad[:len(y_batch)] = y_batch
#             batch.append(y_pad)
#             break
#         start = end
#         end += effective_lenght
#         batch.append(y_batch)

#     batch = np.array(batch)
#     tensors = torch.from_numpy(batch)

#     return tensors

def test_model(test_dataloader,  cfg, checkpoint_path, seed, cv):
    """
    Predicts new data.

    ----
    Args:
        test_data: Path to csv file containing the paths to the audios files for prediction and its labels.

        batch_size: Mini-Batch size.

        checkpoint_path: Path to the file that contains the saved weight of the model trained.

        num_workers: Number of workers to use as paralel processing.

        use_amp: True to use Mixed precision and False otherwise.
    """

    model = Cnn_Model(
        encoder=cfg.train.encoder,
        classes_num=cfg.train.classes_num,
        imagenet_pretrained=False
    )
    model.to(device)

    pred_list = []
    logits_list = []
    labels = []
    labels_hot = []

    print(checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    with torch.no_grad():
        model.eval()

        for test_sample in tqdm(test_dataloader):
            test_audio, test_label = test_sample['image'].to(device), test_sample['target'].to(device)

            out = model(test_audio)

            pred = torch.argmax(out, axis=1).cpu().detach().numpy().tolist()
            label = torch.argmax(test_label, axis=1).cpu().detach().numpy().tolist()

            pred_list.extend(pred)
            labels.extend(label)
            logits_list.extend(out.cpu().detach().numpy().tolist())
            labels_hot.extend(test_label.cpu().detach().numpy().tolist())

    # pred_list = functools.reduce(operator.iconcat, pred_list, [])
    # labels = functools.reduce(operator.iconcat, labels, [])

    # print(metrics.accuracy_score(y_true=labels, y_pred=pred_list))
    # print(metrics.f1_score(y_true=labels, y_pred=pred_list, average='macro'))

    return labels, pred_list, labels_hot, logits_list