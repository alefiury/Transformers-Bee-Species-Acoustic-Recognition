import os
import logging

import librosa
import numpy as np
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# Logger
log = logging.getLogger(__name__)

def audio_file_to_array(batch, cfg):
    """
    Loads the audios from memory

    The audios are loaded from memory and saved in a format that will speed up training afterwards.

    Args:
        batch:
            A huggingface datasets element.

        cfg:
            Hydra config.

    Returns:
        A huggingface datasets element.
    """

    y, _ = librosa.load(
        os.path.join(cfg.data.data_path, batch[cfg.metadata.audio_name_column_name]+".wav"),
        sr=cfg.train.sample_rate,
        mono=True,
        res_type='kaiser_best'
    )

    # Cut the audio in the right segment
    y = y[int(batch[cfg.metadata.begin_time_column_name]*cfg.train.sample_rate) : int(batch[cfg.metadata.end_time_column_name]*cfg.train.sample_rate)]
    batch['image'] = y

    # Creates a hot one encoding version of the label especie_id
    label = np.zeros(cfg.train.classes_num, dtype='f')
    label[batch['especie_id']] = 1
    batch['target'] = label

    # The individual id of each sample is comprised of the name of the file and its begin and end time
    # This id serves as a indicator of each sample
    batch['id'] = batch[cfg.metadata.audio_name_column_name]+'|'+str(batch[cfg.metadata.begin_time_column_name])+'|'+str(batch[cfg.metadata.end_time_column_name])

    return batch

def save_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    train_path: str,
    val_path: str,
    test_path: str,
    cfg: DictConfig
) -> None:

    data_frame_columns_used = ['especie_id', cfg.metadata.audio_name_column_name, cfg.metadata.begin_time_column_name, cfg.metadata.end_time_column_name]

    # Imports a pandas dataframe to a huggingface dataset
    train_dataset = Dataset.from_pandas(X_train[data_frame_columns_used])
    val_dataset = Dataset.from_pandas(X_val[data_frame_columns_used])
    test_dataset = Dataset.from_pandas(X_test[data_frame_columns_used])

    log.info('Loading Audios... ')

    print(cfg.train.sample_rate, cfg.data.preloaded_data_path)

    train_dataset = train_dataset.map(
        audio_file_to_array,
        fn_kwargs={"cfg": cfg},
        remove_columns=train_dataset.column_names,
        num_proc=12
    )

    val_dataset = val_dataset.map(
        audio_file_to_array,
        fn_kwargs={"cfg": cfg},
        remove_columns=val_dataset.column_names,
        num_proc=12
    )

    test_dataset = test_dataset.map(
        audio_file_to_array,
        fn_kwargs={"cfg": cfg},
        remove_columns=test_dataset.column_names,
        num_proc=12
    )

    log.info('Saving Dataset... ')

    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    test_dataset.save_to_disk(test_path)

def prepare_data(
    df: pd.DataFrame,
    seed: int,
    cfg: DictConfig
):
    """
    Preload the audio files.

    The audio files are loaded and saved in disk to accelerate training.

    Args:
        df:
            Pandas dataframe that contains the audio paths and labels.

        seed:
            Split seed.

        original_cwd:
            Path to work directory.

        cfg:
            Hydra config.
    """

    cv1_path = os.path.join(
        cfg.data.preloaded_data_path,
        'seed_' + str(seed),
        'cv1'
    )

    cv2_path = os.path.join(
        cfg.data.preloaded_data_path,
        'seed_' + str(seed),
        'cv2'
    )

    train_cv1_path = os.path.join(cv1_path, 'train')
    val_cv1_path = os.path.join(cv1_path, 'val')
    test_cv1_path = os.path.join(cv1_path, 'test')

    train_cv2_path = os.path.join(cv2_path, 'train')
    val_cv2_path = os.path.join(cv2_path, 'val')
    test_cv2_path = os.path.join(cv2_path, 'test')

    # Create directories if they don't exist for cv 1
    os.makedirs(train_cv1_path, exist_ok=True)
    os.makedirs(val_cv1_path, exist_ok=True)
    os.makedirs(test_cv1_path, exist_ok=True)

    # Create directories if they don't exist for cv 2
    os.makedirs(train_cv2_path, exist_ok=True)
    os.makedirs(val_cv2_path, exist_ok=True)
    os.makedirs(test_cv2_path, exist_ok=True)


    X = df # Contains all columns.
    y = df[['especie_id']] # Dataframe of just the column on which to stratify.

    # CV 1 split
    temp_X_train_cv1, X_test_cv1, temp_y_train_cv1, y_test_cv1 = train_test_split(
        X, y,
        stratify=y,
        test_size=0.5,
        random_state=seed
    )

    X_train_cv1, X_val_cv1, y_train_cv1, y_test_cv1 = train_test_split(
        temp_X_train_cv1, temp_y_train_cv1,
        stratify=temp_y_train_cv1,
        test_size=0.2,
        random_state=seed
    )

    # CV 2 split
    X_test_cv2, temp_X_train_cv2, y_test_cv2, temp_y_train_cv2 = train_test_split(
        X, y,
        stratify=y,
        test_size=0.5,
        random_state=seed
    )

    X_train_cv2, X_val_cv2, y_train_cv2, y_test_cv2 = train_test_split(
        temp_X_train_cv2, temp_y_train_cv2,
        stratify=temp_y_train_cv2,
        test_size=0.2,
        random_state=seed
    )

    assert pd.concat([X_train_cv1, X_val_cv1]).sort_index().equals(X_test_cv2.sort_index())
    assert pd.concat([X_train_cv2, X_val_cv2]).sort_index().equals(X_test_cv1.sort_index())

    # Checks the dataframes shapes to confirm if they are correct
    log.info('Split Shapes CV 1... ')
    log.info(f'Train: {X_train_cv1.shape} | Val: {X_val_cv1.shape} | Test: {X_test_cv1.shape}')

    save_data(
        X_train=X_train_cv1,
        X_val=X_val_cv1,
        X_test=X_test_cv1,
        train_path=train_cv1_path,
        val_path=val_cv1_path,
        test_path=test_cv1_path,
        cfg=cfg
    )

    # Checks the dataframes shapes to confirm if they are correct
    log.info('Split Shapes CV 2... ')
    log.info(f'Train: {X_train_cv1.shape} | Val: {X_val_cv1.shape} | Test: {X_test_cv1.shape}')

    save_data(
        X_train=X_train_cv2,
        X_val=X_val_cv2,
        X_test=X_test_cv2,
        train_path=train_cv2_path,
        val_path=val_cv2_path,
        test_path=test_cv2_path,
        cfg=cfg
    )