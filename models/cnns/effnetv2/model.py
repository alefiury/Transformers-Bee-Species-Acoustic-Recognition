import torch
import torch.nn as nn
import timm
from torchlibrosa.augmentation import SpecAugmentation

from utils.utils import do_mixup

class Cnn_Model(nn.Module):
    def __init__(
        self,
        encoder: str,
        classes_num: int,
        imagenet_pretrained: bool
    ):

        super(Cnn_Model, self).__init__()

        # SpecAugment
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        # Model Encoder (Image model backbone)
        self.encoder = timm.create_model(encoder, pretrained=imagenet_pretrained, num_classes=classes_num)

    def forward(self, input, use_specaug=False):
        """Input : (batch_size, data_length)"""

        # print(type(input))
        input = input.to(dtype=torch.float)

        x = input

        # SpecAugmentation on spectrogram
        if self.training and use_specaug:
            x = self.spec_augmenter(x)

        # Expand to 3 channels because the EffNet model takes 3 channels as input
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3]) # Output shape: (batch size, channels=3, time, frequency)

        x = self.encoder(x)

        x = torch.sigmoid(x)

        return x
