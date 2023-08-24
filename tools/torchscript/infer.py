import torch
import os, sys
from pathlib import Path
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from configs.config import GeneralConfigs
from src.dataloader import LoadDataset
from src.model import init_model

import time

def preprocess_image(image_link, transform, device = "cpu"):
    image = Image.open(image_link).convert("RGB")
    tensor_image = transform(image).unsqueeze(0)
    tensor_image = tensor_image.to(device)
    return tensor_image


def infer():
    general_configs = GeneralConfigs(load_checkpoint="./results/exp5/resnet18_best.pth")

    start_load_model = time.time()
    model = torch.jit.load("./results/exp5/resnet18_best_torchscript.pt")
    device = torch.device('cpu')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model.to(device)
    end_load_model = time.time()

    start_load_image = time.time()
    transform = LoadDataset.preprocess_image_transform(general_configs = general_configs)
    tensor_image = preprocess_image("./test_img/17.png", transform, device)
    end_load_image = time.time()
    
    start_forward = time.time()
    logits = torch.softmax(model(tensor_image), dim=1)
    output = torch.argmax(logits, dim=1)
    end_forward = time.time()
    
    print(logits)
    print(output)
    print(f"duration_load_image: {end_load_image - start_load_image}")
    print(f"duration_load_model: {end_load_model - start_load_model}")
    print(f"duration_forward: {end_forward - start_forward}")


if __name__ == '__main__':
    infer()