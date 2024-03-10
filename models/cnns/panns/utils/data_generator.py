import random
from typing import List, Dict

import torch
import librosa
import torchaudio
import numpy as np
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(
        self,
        batch,
        sample_rate: int = 16000,
        period: int = 5,
        rand_sampling: bool = True,
        apply_norm: bool = False,
        target_dbfs: float = -41.0
    ):

        'Initialization'
        self.batch = batch
        self.sample_rate = sample_rate
        self.period = period
        self.rand_sampling = rand_sampling

        self.apply_norm = apply_norm
        self.target_dbfs = target_dbfs

    def __len__(self):
        return self.batch.num_rows


    def _apply_gain(self, audio: List[float], target_dbfs: float) -> List[float]:
        rms = np.sqrt(np.mean(np.absolute(audio)**2))
        dbfs = 20*np.log10(rms)

        change_in_dBFS_librosa = target_dbfs - dbfs
        db2float = 10 ** (change_in_dBFS_librosa / 20)

        audio_norm = audio*db2float

        audio_norm_clipped = np.clip(audio_norm, -1, 1)

        return audio_norm_clipped


    def _cutorpad(self, audio):
        """
        Cut or pad an audio
        """
        effective_length = self.sample_rate * self.period
        len_audio = audio.shape[0]

        if self.rand_sampling:
            # If audio length is less than wished audio length
            if len_audio < effective_length:
                new_audio = np.zeros(effective_length)
                start = np.random.randint(effective_length - len_audio)
                new_audio[start:start + len_audio] = audio
                audio = new_audio

            # If audio length is bigger than wished audio length
            elif len_audio > effective_length:
                start = np.random.randint(len_audio - effective_length)
                audio = audio[start:start + effective_length]

            # If audio length is equal to wished audio length
            else:
                audio = audio

        else :
            # If audio length is less than wished audio length
            if len_audio < effective_length:
                new_audio = np.zeros(effective_length)
                new_audio[:len_audio] = audio
                audio = new_audio

            # If audio length is bigger than wished audio length
            elif len_audio > effective_length:
                audio = audio[:effective_length]

            # If audio length is equal to wished audio length
            else:
                audio = audio

        return torch.from_numpy(audio).to(dtype=torch.float)

    def __getitem__(self, index):
        audio_original = np.array(self.batch[index]['image'])
        if self.apply_norm:
            audio_original = self._apply_gain(audio_original, self.target_dbfs)
        audio = self._cutorpad(audio_original)
        label = self.batch[index]['target']

        return {
            'image': audio,
            'hot_target': label
        }