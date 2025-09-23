import torch
from torch import optim
from torch.nn import functional as F


def get_optimizer(optimizer_type, training_params, lr, weight_decay):
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(
            params=training_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            params=training_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            params=training_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError('Unknown optimizer type')
    return optimizer

def get_scheduler(scheduler_type, optimizer, epochs, milestones=None, eta_min=None, min_lr=None):
    if scheduler_type == 'cosine':
        if eta_min is not None:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=epochs, eta_min=min_lr
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=epochs
            )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[70, 100], gamma=0.1
        )
    else:
        raise ValueError('Unknown scheduler type')
    return scheduler
