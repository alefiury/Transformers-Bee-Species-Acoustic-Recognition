import os
import glob
import logging
import argparse

import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from omegaconf import OmegaConf
from datasets import load_from_disk

from utils.model_store import model_store_inference
from utils.utils import formatter_single

device = ("cuda" if torch.cuda.is_available() else "cpu")

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-path",
        default="config/config.yaml",
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-m",
        "--choose-model",
        choices=[
            "effnet",
            "panns",
            "psla",
            "ast",
            "mae_ast",
            "ssast"
        ],
        default="effnet",
        help="Choose model to train"
    )

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    # Load a pandas dataframe
    if cfg.data.dataset_csv_path.endswith(".csv"):
        df = pd.read_csv(cfg.data.dataset_csv_path)
    elif cfg.data.dataset_csv_path.endswith(".xlsx"):
        df = pd.read_excel(cfg.data.dataset_csv_path, engine="openpyxl")
    else:
        raise ValueError("File format not supported")
    especie_names = df[cfg.metadata.especies_column_name].unique()
    df["especie_id"] = df[cfg.metadata.especies_column_name].astype("category").cat.codes

    acc_mean = []
    f1_mean = []
    precision_mean = []
    recall_mean = []
    conf_matrixs = []
    for idx, seed in enumerate(cfg.data.seeds):
        for cv in ['cv1', 'cv2']:
            log.info(f'Seed: {seed} | CV: {cv} -> {idx+1}/{len(cfg.data.seeds)}')

            test_path = os.path.join(cfg.data.preloaded_data_path, f'seed_{seed}', cv, 'test')

            preloaded_test_dataset = load_from_disk(test_path)

            preloaded_test_dataset.set_format(type='torch', columns=['image', 'target'])

            labels, pred_list = model_store_inference(
                model_name=args.choose_model,
                test_dataset=preloaded_test_dataset,
                cfg=cfg,
                seed=seed,
                cv=cv
            )

            acc = metrics.accuracy_score(y_true=labels, y_pred=pred_list)
            f1 = metrics.f1_score(y_true=labels, y_pred=pred_list, average='macro')
            precision = metrics.precision_score(y_true=labels, y_pred=pred_list, average='macro')
            recall = metrics.recall_score(y_true=labels, y_pred=pred_list, average='macro')

            conf_martix = metrics.confusion_matrix(y_true=labels, y_pred=pred_list)

            acc_mean.append(acc)
            f1_mean.append(f1)
            precision_mean.append(precision)
            recall_mean.append(recall)
            conf_matrixs.append(conf_martix)

            with open(os.path.join(cfg.scores.scores_path, f'scores_{os.path.basename(cfg.data.output_dir)}.txt'), 'a+') as file:
                file.write(f'Seed: {seed} | '\
                                f'{cv.upper()} | '\
                                f'Accuracy: {acc} | '\
                                f'Precision: {precision} | '\
                                f'Recall: {recall} | '\
                                f'F1 Score: {f1}\n')

    with open(os.path.join(cfg.scores.scores_path, f'scores_{os.path.basename(cfg.data.output_dir)}.txt'), 'a+') as file:
        file.write(f'Accuracy mean: {np.mean(acc_mean)} | '\
                    f'Accuracy std: {np.std(acc_mean)} | '\
                    f'Accuracy median: {np.median(acc_mean)} | '\
                    f'Precision mean: {np.mean(precision_mean)} | '\
                    f'Precision std: {np.std(precision_mean)} | '\
                    f'Precision median: {np.median(precision_mean)} | '\
                    f'Recall mean: {np.mean(recall_mean)} | '\
                    f'Recall std: {np.std(recall_mean)} | '\
                    f'Recall median: {np.median(recall_mean)} | '\
                    f'F1 Mean: {np.mean(f1_mean)} | '\
                    f'F1 std: {np.std(f1_mean)} | '\
                    f'F1 median: {np.median(f1_mean)}')

    np.save(os.path.join(cfg.scores.scores_path, f'{os.path.basename(cfg.data.output_dir)}-conf_matrix.npy'), np.array(conf_matrixs))


if __name__ == '__main__':
    main()