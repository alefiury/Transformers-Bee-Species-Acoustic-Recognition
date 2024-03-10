import warnings

import numpy as np
from torch.optim import lr_scheduler


class CosineWarmupLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup=0, T_max=10):
        """ Description: get warmup cosine lr scheduler
        :param optimizer: (torch.optim.*), torch optimizer
        :param lr_min: (float), minimum learning rate
        :param lr_max: (float), maximum learning rate
        :param warmup: (int), warm up iterations
        :param T_max: (int), maximum number of steps
        Example:
        <<< epochs = 100
        <<< warm_up = 5
        <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
        <<< lrs = []
        <<< for epoch in range(epochs):
        <<< optimizer.step()
        <<< lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        <<< cosine_lr.step()
        <<< plt.plot(lrs, color='r')
        <<< plt.show() """

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup = warmup
        self.T_max = T_max
        self.cur = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warmup == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warmup != 0) & (self.cur <= self.warmup):
            lr = self.lr_min + (self.lr_max - self.lr_min) * self.cur / self.warmup
        else:
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                 (np.cos((self.cur - self.warmup) / (self.T_max - self.warmup) * np.pi) + 1)

        self.cur += 1

        return [lr for _ in self.base_lrs]

class PolynomialLR(lr_scheduler._LRScheduler):
    """
    Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> # xdoctest: +SKIP("undefined vars")
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]