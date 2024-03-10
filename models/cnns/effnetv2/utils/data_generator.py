import random
from typing import List, Dict

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(
        self,
        batch: torch.Tensor,
        sample_rate: int,
        period: int,
        class_num: int,
        use_mixup: bool,
        mixup_alpha: float,
        f_min: int,
        f_max: int,
        n_mfcc: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        data_type: str,
        rand_sampling: bool,
        feature_extractor: str,
        apply_norm: bool,
        target_dbfs: float,
        use_filter_augment: bool
    ):

        'Initialization'
        self.batch = batch
        self.sample_rate = sample_rate
        self.period = period
        self.class_num = class_num

        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.rand_sampling = rand_sampling

        self.f_min = f_min
        self.f_max = f_max

        self.feature_extractor = feature_extractor
        self.n_mfcc = n_mfcc
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.window_size = window_size

        self.data_type = data_type

        self.apply_norm = apply_norm
        self.target_dbfs = target_dbfs

        self.use_filter_augment = use_filter_augment


    def __len__(self):
        return self.batch.num_rows


    # FilterAugment inspired from https://github.com/frednam93/FilterAugSED
    def _filt_aug(self, features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
        if not isinstance(filter_type, str):
            if torch.rand(1).item() < filter_type:
                filter_type = "step"
                n_band = [2, 5]
                min_bw = 4
            else:
                filter_type = "linear"
                n_band = [3, 6]
                min_bw = 6

        batch_size, n_freq_bin, _ = features.shape
        n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
        if n_freq_band > 1:
            while n_freq_bin - n_freq_band * min_bw + 1 < 0:
                min_bw -= 1
            band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                        (n_freq_band - 1,)))[0] + \
                            torch.arange(1, n_freq_band) * min_bw
            band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

            if filter_type == "step":
                band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
                band_factors = 10 ** (band_factors / 20)

                freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
                for i in range(n_freq_band):
                    freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

            elif filter_type == "linear":
                band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
                freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
                for i in range(n_freq_band):
                    for j in range(batch_size):
                        freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                            torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                        band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
                freq_filt = 10 ** (freq_filt / 20)
            return features * freq_filt

        else:
            return features


    def _cutorpad(self, audio: List[float]) -> List[float]:
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

        return audio


    def _apply_gain(self, audio: List[float], target_dbfs: float) -> List[float]:
        rms = np.sqrt(np.mean(np.absolute(audio)**2))
        dbfs = 20*np.log10(rms)

        change_in_dBFS_librosa = target_dbfs - dbfs
        db2float = 10 ** (change_in_dBFS_librosa / 20)

        audio_norm = audio*db2float

        audio_norm_clipped = np.clip(audio_norm, -1, 1)

        return audio_norm_clipped


    def __getitem__(self, index: int) -> Dict:

        if self.use_mixup and self.data_type=='train':
            # Samples a random audio from the dataset to do mixup
            rand_index = random.randint(0, self.batch.num_rows-1)

            # Makes sure that the class from the random audio is different than the main audio
            while np.argmax(self.batch[rand_index]['target']) == np.argmax(self.batch[index]['target']):
                rand_index = random.randint(0, self.batch.num_rows-1)

            audio_original = np.array(self.batch[index]['image'])
            audio_rand = np.array(self.batch[rand_index]['image'])

            if self.apply_norm:
                audio_original = self._apply_gain(audio_original, self.target_dbfs)
                audio_rand = self._apply_gain(audio_rand, self.target_dbfs)

            # Cut audios
            audio_original = self._cutorpad(audio_original)
            audio_rand = self._cutorpad(audio_rand)

            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Do mixup
            audio = mix_lambda * audio_original + (1 - mix_lambda) * audio_rand

            # Change targets
            label = np.zeros(self.class_num, dtype='f')
            label[np.argmax(self.batch[index]['target'])] = mix_lambda
            label[np.argmax(self.batch[rand_index]['target'])] = 1 - mix_lambda

        else:
            audio_original = np.array(self.batch[index]['image'])
            if self.apply_norm:
                audio_original = self._apply_gain(audio_original, self.target_dbfs)
            audio = self._cutorpad(audio_original)
            label = self.batch[index]['target']

        if self.feature_extractor == 'mfcc':
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.window_size,
                hop_length=self.hop_size,
                n_mels=self.mel_bins,
                fmin=self.f_min,
                fmax=self.f_max
            )

            features = mfcc

        elif self.feature_extractor == 'log_melspec':
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.window_size,
                hop_length=self.hop_size,
                n_mels=self.mel_bins,
                fmin=self.f_min,
                fmax=self.f_max
            )

            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

            features = log_mel_spectrogram

        else:
            print("Invalid feature, exiting... ")
            exit()

        if self.use_filter_augment and self.data_type=='train':
            tensor_signal = torch.tensor(features).unsqueeze(0)
            features = self._filt_aug(tensor_signal).numpy().squeeze(0)

        return {
            'image': np.expand_dims(features, axis=0),
            'target': label
        }