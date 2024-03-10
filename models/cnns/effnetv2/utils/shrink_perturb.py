import sys
from copy import deepcopy

import torch

from model import Cnn_Model

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def shrink_perturb(model=None, shrink_alpha: float = 0.4, perturb_gama: float = 0.1, cfg=None):
    # using a randomly-initialized model as a noise source respects how different kinds 
    # of parameters are often initialized differently
    new_init = Cnn_Model(
        encoder=cfg.train.encoder,
        classes_num=cfg.train.classes_num,
        imagenet_pretrained=False
    )
    new_init.to(device)

    params1 = new_init.parameters()
    params2 = model.parameters()
    for p1, p2 in zip(*[params1, params2]):
        p1.data = deepcopy(shrink_alpha * p2.data + perturb_gama * p1.data)
    del model
    return new_init