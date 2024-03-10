import random
from typing import List, Dict

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

class DataGenerator(Dataset):
    def __init__(
        self,
        batch: dict,
        sample_rate: int,
        period: int,
        class_num: int,
        use_mixup: bool,
        mixup_alpha: float,
        data_type: str,
        freqm: int,
        timem: int,
        skip_spec_norm: bool,
        spec_norm_mean: float,
        spec_norm_std: float,
        use_specaug: bool,
        padding_type: bool,
        apply_audio_norm: bool,
        target_dbfs: float,
        target_length: int,
        use_filter_augment: bool
    ):

        """Initialization"""
        self.batch = batch

        self.sample_rate = sample_rate
        self.period = period
        self.data_type = data_type

        self.spec_norm_std = spec_norm_std
        self.spec_norm_mean = spec_norm_mean
        self.skip_spec_norm = skip_spec_norm

        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.use_specaug = use_specaug
        self.freqm = freqm
        self.timem = timem

        self.class_num = class_num
        self.target_length = target_length

        self.padding_type = padding_type

        self.apply_audio_norm = apply_audio_norm
        self.target_dbfs = target_dbfs

        self.use_filter_augment = use_filter_augment

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


    def _cutorpad(self, audio: List[float]) -> List[float]:
        """
        Cut or pad an audio
        """
        effective_length = self.sample_rate * self.period
        len_audio = audio.shape[0]

        if self.padding_type=="rand_trunc":
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

        elif  self.padding_type=="fixed_trunc":
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

        elif  self.padding_type=="repeat":
            # If audio length is less than wished audio length
            if len_audio < effective_length:
                n_repeat = int(effective_length/len_audio)
                audio = np.tile(audio, n_repeat)
                audio = np.pad(
                    audio,
                    (0, effective_length - len(audio)),
                    mode="constant",
                    constant_values=0,
                )
            # If audio length is bigger than wished audio length
            elif len_audio > effective_length:
                start = np.random.randint(len_audio - effective_length)
                audio = audio[start:start + effective_length]

            # If audio length is equal to wished audio length
            else:
                audio = audio

        else:
            raise NotImplementedError(
                f"data_filling {self.padding_type} not implemented"
            )

        # Expand one dimension related to the channel dimension
        return torch.from_numpy(audio).unsqueeze(0)

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



    def __getitem__(self, index: int) -> Dict[List[float], List[int]]:

        if self.use_mixup and self.data_type=='train':
            # Samples a random audio from the dataset to do mixup
            rand_index = random.randint(0, self.batch.num_rows-1)

            # Makes sure that the class from the random audio is different than the main audio
            while np.argmax(self.batch[rand_index]['target']) == np.argmax(self.batch[index]['target']):
                rand_index = random.randint(0, self.batch.num_rows-1)

            audio_original = np.array(self.batch[index]['image'])
            audio_rand = np.array(self.batch[rand_index]['image'])

            if self.apply_audio_norm:
                audio_original = self._apply_gain(audio_original, self.target_dbfs)
                audio_rand = self._apply_gain(audio_rand, self.target_dbfs)

            # Cut audios
            audio_original = self._cutorpad(audio_original)
            audio_rand = self._cutorpad(audio_rand)

            # Normalize data as in: https://github.com/YuanGongND/ast/blob/master/src/dataloader.py
            audio_original = audio_original - audio_original.mean()
            audio_rand = audio_rand - audio_rand.mean()

            # Sample lambda from a beta distribution based on the value of alpha
            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Do mixup
            audio = mix_lambda * audio_original + (1 - mix_lambda) * audio_rand

            # Change targets
            label = np.zeros(self.class_num, dtype='f')
            label[np.argmax(self.batch[index]['target'])] = mix_lambda
            label[np.argmax(self.batch[rand_index]['target'])] = 1 - mix_lambda

        else:
            audio_original = np.array(self.batch[index]['image'])
            if self.apply_audio_norm:
                audio_original = self._apply_gain(audio_original, self.target_dbfs)
            audio = self._cutorpad(audio_original)
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
            frame_shift=10
        )

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if self.use_filter_augment and self.data_type=='train':
            fbank = fbank.unsqueeze(0)
            fbank = self._filt_aug(fbank).squeeze(0)

        # SpecAugmentation
        if self.use_specaug and self.data_type=='train':
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

        # normalize the input for both training and test
        if not self.skip_spec_norm:
            fbank = (fbank - self.spec_norm_mean) / (self.spec_norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        return {
            'image': fbank, # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
            'target': label
        }