import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

from configs.config import LogConfigs, GeneralConfigs, TrainConfigs, ValidConfigs
from src.utils import logger_print


class ClsfModel(nn.Module):
    def __init__(self):
        super(ClsfModel, self).__init__()
        pass
    def forward(self, x):
        return x


def init_model(general_configs: GeneralConfigs):
    if general_configs.model_backbone == 'custom':
        model = ClsfModel()
    else:
        if general_configs.model_backbone == 'efficientnetv2s':
            backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        elif general_configs.model_backbone == 'mobilenetv3s':
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        elif general_configs.model_backbone == 'regnetx800mf':
            backbone = regnet_x_800mf(weights=RegNet_X_800MF_Weights.DEFAULT)

        elif general_configs.model_backbone == 'regnetx8gf':
            backbone = regnet_x_8gf(weights=RegNet_X_8GF_Weights.DEFAULT)

        elif general_configs.model_backbone == 'regnety32gf':
            backbone = regnet_y_32gf(weights=RegNet_Y_32GF_Weights.DEFAULT)

        elif general_configs.model_backbone == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
           
        elif general_configs.model_backbone == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        elif general_configs.model_backbone == 'resnet152':
            backbone = resnet152(weights=ResNet152_Weights.DEFAULT)

        elif general_configs.model_backbone == 'squeezenet1_1':
            backbone = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)

        model = nn.Sequential(
            backbone,
            nn.Linear(1000, general_configs.class_num),
        )
    
    if general_configs.load_checkpoint is not None:
        model.load_state_dict(torch.load(general_configs.load_checkpoint))
        logger_print(general_configs.logger, f"""Load checkpoint from {general_configs.load_checkpoint} successfully""")

    return model
