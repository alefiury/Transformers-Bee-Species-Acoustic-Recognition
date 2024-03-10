import torch
import torch.nn as nn

class PANNsLoss(nn.Module):
    """
    Loss that uses the BCELoss, dealing with cases where
    the probabilities are infinite (inf) or unknown (nan)
    """
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = torch.sigmoid(input)
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_).float()

        target = target.float()

        return self.bce(input_, target)