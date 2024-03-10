import random
from typing import Dict, Optional

import torch
import librosa
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


class DataGenerator(Dataset):
    def __init__(
        self,
        batch: torch.Tensor,
        sample_rate: int,
        period: int,
        type: str,
        freqm: Optional[int],
        timem: Optional[int],
        skip_norm: bool,
        norm_mean: Optional[float],
        norm_std: Optional[float],
        mixup_alpha: Optional[float],
        use_audio_mixup: bool,
        use_specaug: bool,
        rand_sampling: bool,
        insert_noise: bool,
        class_num: int,
        target_length: int
    ):

        'Initialization'
        self.batch = batch

        self.sample_rate = sample_rate
        self.period = period

        self.type = type

        self.freqm = freqm
        self.timem = timem

        self.skip_norm = skip_norm
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.mixup_alpha = mixup_alpha
        self.use_audio_mixup = use_audio_mixup
        self.use_specaug = use_specaug
        self.rand_sampling = rand_sampling
        self.insert_noise = insert_noise

        self.class_num = class_num
        self.target_length = target_length


    def __len__(self):
        return len(self.batch)

    def _cutorpad(self, audio: np.ndarray) -> torch.Tensor:
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

        # Expand one dimension related to the channel dimension
        return torch.from_numpy(audio).expand(1, audio.shape[0])

    def __getitem__(self, index: int) -> Dict[torch.Tensor, np.ndarray]:

        if self.use_audio_mixup and self.type=='train':

            # Samples a random audio from the dataset to do mixup
            rand_index = random.randint(0, self.batch.num_rows-1)

            # Makes sure that the class from the random audio is different than the main audio
            while np.argmax(self.batch[rand_index]['target']) == np.argmax(self.batch[index]['target']):
                rand_index = random.randint(0, self.batch.num_rows-1)

            audio1 = np.array(self.batch[index]['image'])
            audio2 = np.array(self.batch[rand_index]['image'])

            audio1 = self._cutorpad(audio1)
            audio2 = self._cutorpad(audio2)

            # Normalize data as in: https://github.com/YuanGongND/ast/blob/master/src/dataloader.py
            audio1 = audio1 - audio1.mean()
            audio2 = audio2 - audio2.mean()

            # Sample lambda from a beta distribution based on the value of alpha
            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Do mixup
            audio = mix_lambda * audio1 + (1 - mix_lambda) * audio2

            label = np.zeros(self.class_num, dtype='f')
            label[np.argmax(self.batch[index]['target'])] = mix_lambda
            label[np.argmax(self.batch[rand_index]['target'])] = 1 - mix_lambda
        else:
            audio = np.array(self.batch[index]['image'])
            audio = self._cutorpad(audio)
            label = self.batch[index]['target']

        # Normalize data as in: https://github.com/YuanGongND/ast/blob/master/src/dataloader.py
        audio = audio - audio.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            audio,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10.0
        )

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # Cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        # SpecAugmentation
        if self.use_specaug and self.type=='train':
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            # Transpose because the FrequencyMasking and TimeMasking
            # from pytorch needs a dimension format as: (…, freq, time)
            fbank = torch.transpose(fbank, 0, 1)
            # Satisfies new torchaudio version, which only accept [1, freq, time]
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            # Satisfies new torchaudio version, which only accept [1, freq, time]
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # Normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # Skip normalization if you are trying to get the normalization stats
        else:
            pass

        if self.insert_noise:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        return {
            'image': fbank.to(dtype=torch.float),
            'hot_target': label
        }