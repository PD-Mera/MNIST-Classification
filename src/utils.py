import argparse
import os
import yaml
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import logging

from configs.config import LogConfigs, GeneralConfigs, TrainConfigs, ValidConfigs


def load_cfg(cfg_path):
    with open(cfg_path, mode='r') as f:
        yaml_data = f.read()

    data = yaml.load(yaml_data, Loader=yaml.Loader)
    return data

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/config.yaml', help='cfg.yaml path')
    return parser.parse_args()


def logger_print(logger: logging.Logger, message: str):
    try:
        logger.info(message)
    except:
        print(message)


def is_image_file(filename: str):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def create_confusion_matrix(num_class: int):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.zeros((num_class, num_class)).to(device)


def update_confusion_matrix(confusion_matrix, predict_batch, target_batch):
    predict_batch = predict_batch.squeeze(1)
    target_batch = target_batch#.squeeze(1)
    for i in range(target_batch.size(0)):
        confusion_matrix[predict_batch[i]][target_batch[i]] += 1
    return confusion_matrix


def draw_confusion_matrix(general_configs: GeneralConfigs, confusion_matrix, exp_save_dir):
    df_cm = pd.DataFrame(confusion_matrix, index=general_configs.list_class_names, columns=general_configs.list_class_names)
    sn.set(font_scale=1.4) # for label size
    sn.set(rc={'figure.figsize':(general_configs.class_num + 9, general_configs.class_num + 9)})
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Oranges', fmt='.0f') # font size
    plt.title('Confusion Matrix', fontsize = 20, pad = 10)
    plt.xlabel('Target', fontsize = 16) # x-axis label with fontsize 16
    plt.ylabel('Predict', fontsize = 16) # x-axis label with fontsize 16
    plt.tight_layout()
    plt.savefig(exp_save_dir + "/" + general_configs.model_backbone + "_confusion_matrix.jpg")
    plt.close()


def create_exp_save_dir(save_model_dir: str):
    max_exp = 0
    for exp_folder in os.listdir(save_model_dir):
        exp_folder: str
        if "exp" not in exp_folder:
            continue
        if int(exp_folder.replace("exp", "")) > max_exp:
            max_exp = int(exp_folder.replace("exp", ""))
    
    max_exp += 1
    return os.path.join(save_model_dir, f"exp{max_exp}")