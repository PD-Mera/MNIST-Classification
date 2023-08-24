from typing import Union

import torch
from torch import nn

from configs.config import TrainConfigs, ValidConfigs

def init_optimizer(model: nn.Module, train_configs: TrainConfigs):
    optim_name = train_configs.optimizer
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_configs.learning_rate)

    return optimizer
