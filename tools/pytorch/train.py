import time, datetime, math
import torch
from torch.utils.data import DataLoader
import os, sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from configs.config import LogConfigs, GeneralConfigs, TrainConfigs, ValidConfigs
from src.dataloader import LoadDataset
from src.model import init_model
from src.optimizer import init_optimizer
from src.loss import init_loss
from src.utils import create_confusion_matrix, update_confusion_matrix, draw_confusion_matrix, create_exp_save_dir


def train():
    general_configs = GeneralConfigs(load_checkpoint="./results/exp4/resnet18_best.pth")

    os.makedirs(general_configs.save_model_dir, exist_ok=True)
    exp_save_dir = create_exp_save_dir(general_configs.save_model_dir)
    os.makedirs(exp_save_dir)

    general_configs.logger = LogConfigs(experiment_dir = exp_save_dir).initialize()
    train_configs = TrainConfigs(data_path = "./data/mnist_png/training",
                                 learning_rate=1e-5)
    valid_configs = ValidConfigs(data_path = "./data/mnist_png/testing")
    

    train_data = LoadDataset(general_configs=general_configs,
                             runtime_configs=train_configs)
    valid_data = LoadDataset(general_configs=general_configs,
                             runtime_configs=valid_configs)
    
    train_loader = DataLoader(train_data, 
                              batch_size = train_configs.batch_size, 
                              num_workers = train_configs.num_worker, shuffle=True)
    valid_loader = DataLoader(valid_data, 
                              batch_size=valid_configs.batch_size, 
                              num_workers=valid_configs.num_worker)
    
    model = init_model(general_configs)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = init_optimizer(model, train_configs)
    loss_fn = init_loss(general_configs, train_configs, train_data)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    highest_acc = 0

    batch_number = math.ceil(len(train_loader.dataset) / train_configs.batch_size)
    general_configs.logger.info("Start training loop")
    for epoch in range(train_configs.num_epoch):
        # Start Training
        model.train()
        tik = time.time()
        train_correct = 0
        c_matrix = create_confusion_matrix(general_configs.class_num)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            target = target
            train_correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_number // 10 > 0:
                print_fre = batch_number // 10
            else:
                print_fre = 1
            if batch_idx % print_fre == print_fre - 1:
                iter_num = batch_idx * len(data)
                total_data = len(train_loader.dataset)
                iter_num = str(iter_num).zfill(len(str(total_data)))
                total_percent = 100. * batch_idx / len(train_loader)
                general_configs.logger.info(f'Train Epoch {epoch + 1}: [{iter_num}/{total_data} ({total_percent:2.0f}%)] | Loss: {loss.item():.10f}')
                

        # Start Validating
        general_configs.logger.info(f"Validating {len(valid_loader.dataset)} images")
        model.eval()
        valid_correct = 0

        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            target = target
            valid_correct += pred.eq(target.view_as(pred)).sum().item()
            c_matrix = update_confusion_matrix(c_matrix, pred, target)


        general_configs.logger.info('Validation completed\n')

        train_accuracy = 100. * train_correct / len(train_loader.dataset)
        valid_accuracy = 100. * valid_correct / len(valid_loader.dataset)
        general_configs.logger.info('Train set: Accuracy: {}/{} ({:.2f}%)'.format(
            train_correct, len(train_loader.dataset), train_accuracy))
        general_configs.logger.info('Valid set: Accuracy: {}/{} ({:.2f}%)'.format(
            valid_correct, len(valid_loader.dataset), valid_accuracy))
        general_configs.logger.info(f'\nConfusion matrix of valid set\n{c_matrix}')

        
        tok = time.time()
        runtime = tok - tik
        eta = int(runtime * (train_configs.num_epoch - epoch - 1))
        eta = str(datetime.timedelta(seconds=eta))
        general_configs.logger.info(f'Runing time: Epoch {epoch + 1}: {str(datetime.timedelta(seconds=int(runtime)))} | ETA: {eta}')
        
        
        torch.save(model.state_dict(), os.path.join(exp_save_dir, f'{general_configs.model_backbone}_last.pth'))
        general_configs.logger.info(f"Saving last model to {os.path.join(exp_save_dir, f'{general_configs.model_backbone}_last.pth')}\n")

        if train_accuracy >= highest_acc:
            highest_acc = train_accuracy
            torch.save(model.state_dict(), os.path.join(exp_save_dir, f'{general_configs.model_backbone}_best.pth'))
            general_configs.logger.info(f"Saving best model to {os.path.join(exp_save_dir, f'{general_configs.model_backbone}_best.pth')}\n")
            draw_confusion_matrix(general_configs, c_matrix.cpu().numpy(), exp_save_dir)

if __name__ == '__main__':
    train()