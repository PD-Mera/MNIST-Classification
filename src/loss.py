import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Union

from configs.config import TrainConfigs, GeneralConfigs
from src.dataloader import LoadDataset
from src.utils import logger_print

def calculate_weights(general_configs: GeneralConfigs, train_dataset: LoadDataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    count_label = train_dataset.count_label
    with open(general_configs.classnames_file) as f:
        class_names = f.read().split("\n")

    weights = [count_label[x] for x in class_names]

    total_datasize = sum(weights)
    weights = [(1.0 - (x / total_datasize)) for x in weights]
    
    logger_print(general_configs.logger, f"""Init loss with weight {weights} successfully""")
    return torch.tensor(weights, dtype=torch.float32).to(device)


class ClsfLoss(nn.Module):
    def __init__(self):
        super(ClsfLoss, self).__init__()
        pass
    def forward(self, x):
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def init_loss(general_configs: GeneralConfigs, train_configs: TrainConfigs, train_dataset: LoadDataset):
    if train_configs.loss_fn == 'custom':
        loss = ClsfLoss()
    elif train_configs.loss_fn == 'CE':
        weights = calculate_weights(general_configs, train_dataset)
        loss = nn.CrossEntropyLoss(weight = weights)
    elif train_configs.loss_fn == 'FocalLoss':
        loss = FocalLoss(gamma=2)
    return loss

