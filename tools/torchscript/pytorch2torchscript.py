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

def pytorch2torchscript():
    checkpoint_path = "./results/exp5/resnet18_best.pth"
    general_configs = GeneralConfigs(load_checkpoint = checkpoint_path)
    model = init_model(general_configs)
    model.eval()

    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    save_torchscript_path = checkpoint_path.replace(".pth", "_torchscript.pt")
    traced_script_module.save(save_torchscript_path)
    print(f"Pytorch model to Torch Script converted!!! Model save to {save_torchscript_path}")


if __name__ == '__main__':
    pytorch2torchscript()    