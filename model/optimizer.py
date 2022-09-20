import math
import warnings

import torch.optim as optimizer_zoo
import logging
import torch
from torch.optim.lr_scheduler import _LRScheduler


def build_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optimizer_zoo.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )

    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optimizer_zoo.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR
        )

    return optimizer


def build_lr_scheduler(cfg, optimizer, **kwargs):
    logger = logging.getLogger(__name__)
    if cfg.TRAIN.LR_SCHEDULER == 'MultiStepLR':
        last_epoch = kwargs["last_epoch"] if 'last_epoch' in kwargs else -1
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.TRAIN.MILESTONES,
                                                            gamma=cfg.TRAIN.GAMMA,
                                                            last_epoch=last_epoch)

        print(
            "=> Use MultiStepLR. MILESTONES : {}. GAMMA : {}. last_epoch : {}".format(cfg.TRAIN.MILESTONES,
                                                                                      cfg.TRAIN.GAMMA, last_epoch))
    elif cfg.TRAIN.LR_SCHEDULER == 'CosineAnnealingLR':
        num_iters_per_epoch = kwargs["num_iters_per_epoch"]

        max_epochs = cfg.TRAIN.EPOCHS + cfg.TRAIN.WARMUP_EPOCHS
        max_steps = max_epochs * num_iters_per_epoch

        warmup_epochs = cfg.TRAIN.WARMUP_EPOCHS
        warmup_steps = warmup_epochs * num_iters_per_epoch

        last_epoch = kwargs["last_epoch"] if 'last_epoch' in kwargs else -1
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                     warmup_steps,
                                                     max_steps,
                                                     last_epoch=last_epoch
                                                     )
        print(
            "=> Use CosineAnnealingLR. warmup_steps: {} max_steps : {}. last_epoch : {}".format(warmup_steps,
                                                                                                max_steps, last_epoch))
    else:
        logger.error("Please Check if LR_SCHEDULER is valid")
        raise Exception("Please Check if LR_SCHEDULER is valid")

    return lr_scheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
            self,
            optimizer,
            warmup_epochs,
            max_epochs,
            warmup_start_lr=0.0,
            eta_min=1e-8,
            last_epoch=-1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                    1 +
                    math.cos(
                        math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
