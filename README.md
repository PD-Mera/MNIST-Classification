# MNIST Classification

This project fine-tune torchvision pretrained weights to classify MNIST images and try to run Pytorch model with Rust

![sample](./assets/sample.jpg "sample")

## Environments

- Python 3.8.10
- torch 2.0.1
- torchvision 0.15.2

Install requirements

``` bash
pip install -U pip
pip install -r requirements.txt
```

## Data

You can download MNIST dataset from [here](https://drive.google.com/file/d/1JimUxm4tpbsg2zOqbGnbJDzJHwgp6by7/view?usp=share_link), use torch dataset or simply run this script

``` bash
bash scripts/download_data.sh
```

## Config

List all datafolder in `./cfg/config.yaml`

Modify class names in `./cfg/classs.names`

Modify config in `./cfg/config.yaml` or create your own `.yaml` config file with the same format.

## Train

Simply run 

``` bash
python tools/train.py --cfg ./cfg/config.yaml
```

## Experiment Results

Some experiment results

| Model | Accuracy | Confusion Matrix | Pretrained | Model size |
| --- | :---: | :---: | :---: | :---: |
| **Resnet18** | 100.00% (In train set) | ![CM1](./assets/resnet18_confusion_matrix.jpg "CM1 Image") | [Model](https://bit.ly/45LcotQ) (Private) | 44.70MB |

You can download weight file above and put in `weights` folder and run inference with app

``` bash
uvicorn tools.app:app --host 127.0.0.1 --port 12345
```

## Infer

You can infer with

``` bash
python tools/infer.py --cfg ./cfg/config.yaml
```

## Convert to other format

### ONNX

Refer [here](./onnx/README.md)

## Some inference results

You can try on your own :wink:

## Reference

- [PD-Mera/Document-Checker](https://github.com/PD-Mera/Document-Checker)