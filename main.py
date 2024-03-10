import os
import logging
import argparse

import torch
import wandb
import pandas as pd
from omegaconf import OmegaConf
from datasets import load_from_disk

from utils.model_store import model_store_training
from utils.utils import formatter_single

device = ("cuda" if torch.cuda.is_available() else "cpu")

wandb.login()

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)


def main() -> None:
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

    print(especie_names)
    print(len(especie_names))
    print(df[cfg.metadata.especies_column_name].value_counts())

    print(df[[cfg.metadata.especies_column_name, "especie_id", "genero"]].value_counts().reset_index(name="count"))

    wandb_config = {
        **cfg.train,
        **cfg.data,
        **cfg.metadata,
        **cfg.test
    }

    seeds = cfg.train.seeds

    for idx, seed in enumerate(seeds):
        for cv in ['cv1', 'cv2']:
            log.info(f'Seed: {seed} | CV: {cv} -> {idx+1}/{len(cfg.train.seeds)}')

            # Start a new run
            run = wandb.init(
                project=f'{os.path.basename(cfg.data.output_dir)}',
                config=wandb_config,
                reinit=True
            )

            # Change run name to facilitate train statistics traceability
            wandb.run.name = f'{seed}-{cv}'
            wandb.run.save()

            train_path = os.path.join(
                cfg.data.preloaded_data_path,
                f'seed_{seed}',
                cv,
                'train'
            )

            val_path = os.path.join(
                cfg.data.preloaded_data_path,
                f'seed_{seed}',
                cv,
                'val'
            )

            preloaded_train_dataset = load_from_disk(train_path)
            preloaded_val_dataset = load_from_disk(val_path)

            preloaded_train_dataset.set_format(type="torch", columns=["image", "target"])
            preloaded_val_dataset.set_format(type="torch", columns=["image", "target"])

            # Train model
            model_store_training(
                model_name=args.choose_model,
                train_dataset=preloaded_train_dataset,
                val_dataset=preloaded_val_dataset,
                cfg=cfg,
                seed=seed,
                cv=cv
            )

            # Finish a new run
            run.finish()


if __name__ == '__main__':
    main()