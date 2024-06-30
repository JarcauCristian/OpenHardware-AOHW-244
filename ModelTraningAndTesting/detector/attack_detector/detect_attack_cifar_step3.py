# TODO CROP files
# TODO save onnx


import os.path
from datetime import datetime
from typing import Optional, Tuple

import torch
from torch import optim, nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, \
    efficientnet_v2_l, EfficientNet_V2_L_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.transforms import transforms
from tqdm import tqdm
import logging

import sys
import os

from utils.datasets import DiffDataset
from utils.log_utils import setup_logger
from utils.stats import compute_mean_std

sys.path.append(os.getcwd())  # unlikely to be needed

from utils.timing import timeit
from utils.conversions import get_optimizer_name, get_model_name, get_scheduler_name
from utils.environment_utils import is_linux_os, can_compile_torch_model, get_machine_name


def get_dataset(path: str, difference: int, mean: torch.Tensor, std: torch.Tensor) -> ImageFolder:

    transformations = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = ImageFolder(path, transform=transformations)

    if difference > 0:
        dataset = DiffDataset(dataset, difference=difference)

    return dataset


def get_loader(path: str, batch_size: int, shuffle: bool, difference: int, mean: torch.Tensor, std: torch.Tensor) -> DataLoader:
    dataset = get_dataset(path, mean=mean, std=std, difference=difference)
    num_workers = 16 if is_linux_os() else 1
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dataloader


@torch.no_grad()
def get_loss(model: nn.Module,
             device: str,
             criterion: nn.Module,
             path: str,
             batch_size: int,
             mean: torch.Tensor,
             std: torch.Tensor,
             difference: int) -> float:
    model.eval()
    loader = get_loader(path, batch_size=batch_size, shuffle=False, mean=mean, std=std, difference=difference)
    total_loss = 0
    n_items = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        predicted = model(data)
        loss = criterion(predicted, labels)
        total_loss += loss.detach().cpu().item()
        n_items += len(data)
    return total_loss / n_items


def get_mean_std_train(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    transformations = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(path, transform=transformations)
    return compute_mean_std(dataset)


def save_onnx(model, path, device):
    model.eval()
    fake_input = torch.randn(1, 3, 224, 224).to(device)
    # onnx_program = torch.onnx.dynamo_export(model, fake_input)
    # onnx_program.save(path)

    # source: https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
    torch.onnx.export(model,  # model being run
                      fake_input,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})

    # input_names = ["actual_input"]
    # output_names = ["output"]


def save_model(model: nn.Module, path: str, epoch: int, train_accuracy: float, test_accuracy: float, device: str) -> None:
    # save checkpoint
    path_pth = f'{path}/epoch={epoch}_trainacc={train_accuracy}_testacc={test_accuracy}.pth'
    torch.save(model.state_dict(), path_pth)

    # save onnx file
    path_onnx = f'{path}/epoch={epoch}_trainacc={train_accuracy}_testacc={test_accuracy}.onnx'
    save_onnx(model, path_onnx, device)


@timeit
def train_model(model: nn.Module,
                device: str,
                train_path: str,
                test_path: str,
                epochs: int,
                batch_size: int,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[LRScheduler],
                writer: SummaryWriter,
                save_to: str,
                log_interval: int = 1,
                difference: int = 0):


    model.train()
    save_model(model, save_to, epoch=0, train_accuracy=0, test_accuracy=0, device=device)

    logging.info('Getting mean and std...')
    mean, std = get_mean_std_train(train_path)

    train_loader = get_loader(train_path, batch_size=batch_size, shuffle=True, difference=difference, mean=mean, std=std)
    criterion = CrossEntropyLoss()


    for epoch in tqdm(range(1, epochs + 1), 'Training', initial=1):
        model.train()
        total_loss = 0
        n_items = 0
        for data, labels in train_loader:
            model.zero_grad()
            data, labels = data.to(device), labels.to(device)
            predicted = model(data)
            loss = criterion(predicted, labels)
            total_loss += loss.item()
            n_items += len(data)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
            logging.info(f'scheduled lr={scheduler.get_last_lr()}')
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        logging.info(f'Average running loss at epoch {epoch}/{epochs}: {total_loss / n_items}')
        writer.add_scalar('Running loss/Train', total_loss / n_items, epoch)

        accurate_train_loss = get_loss(model, device, criterion, train_path, batch_size, mean=mean, std=std, difference=difference)
        logging.info(f'Average train loss at epoch {epoch}/{epochs}: {accurate_train_loss}')
        writer.add_scalar('Loss/Train', accurate_train_loss, epoch)

        if epoch % log_interval == 0:
            train_accuracy = test_model(model, device, train_path, batch_size, mean=mean, std=std, difference=difference)
            writer.add_scalar('Train accuracy/Train', train_accuracy, epoch)
            logging.info(f'Train accuracy/Train at epoch {epoch}: {train_accuracy}')

            test_accuracy = test_model(model, device, test_path, batch_size, mean=mean, std=std, difference=difference)
            writer.add_scalar('Test accuracy/Train', test_accuracy, epoch)
            logging.info(f'Test accuracy/Train at epoch {epoch}: {test_accuracy}')

            save_model(model, save_to, epoch, train_accuracy, test_accuracy, device)


@torch.no_grad()
def test_model(model: nn.Module, device: str, test_path: str, batch_size: int, mean: torch.Tensor, std: torch.Tensor,
               difference: int = 0) -> float:
    model.eval()
    loader = get_loader(test_path, batch_size=batch_size, shuffle=False, difference=difference, mean=mean, std=std)
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        prob = model(images)
        y_hat = prob.argmax(dim=-1)
        correct += torch.sum(labels == y_hat).item()
        total += len(images)
    test_accuracy = correct / total
    return test_accuracy


@torch.no_grad()
def get_model(device: str, model_type: str, dilation: Optional[int] = None) -> nn.Module:
    assert model_type is not None, f'Got {model_type=}'
    allowed_models = ['resnet18', 'resnet34', 'resnet50', 'efficientnet_v2_l', 'efficientnet_v2_m']
    model_type = model_type.lower()
    assert model_type in allowed_models, f'Model {model_type} not in {allowed_models}'
    _model = None
    if model_type == 'resnet18':
        _model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_type == 'resnet34':
        _model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif model_type == 'resnet50':
        _model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_type == 'efficientnet_v2_l':
        _model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    elif model_type == 'efficientnet_v2_m':
        _model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    assert _model is not None

    dilation = 1 if dilation is None or dilation < 1 else dilation
    # Modify conv1 to suit CIFAR-10/CIFAR-100
    # _model.conv1 = nn.Conv2d(3, 64,
    #                          kernel_size=3,
    #                          stride=1,
    #                          padding=1,
    #                          bias=False,
    #                          dilation=dilation)

    if model_type.startswith('resnet'):
        _model.fc = torch.nn.Linear(in_features=_model.fc.in_features, out_features=2)
    elif model_type.startswith('efficientnet'):
        _model.classifier[1] = nn.Linear(_model.classifier[1].in_features, 2)

    _model.to(device)
    return _model


@timeit
def main():
    machine_name = get_machine_name()
    logging.info(f'Working on {machine_name=}')

    use_cuda: bool = True
    device: str = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    logging.info(f'{device=}')

    root_images_attack = '../data/german/detect_attack'

    difference = 0  # 0 = original dataset; 1 = first order difference; 2 = second order difference

    # experiment_dir = 'ResNet18_attacker=LinfFastGradient_epsilon=0.03'  # FGSM, works fine genuine and 1st, 2nd order difference
    # experiment_dir = 'cifar100_ResNet18_attacker=LinfFastGradient_epsilon=0.03'

    # experiment_dir = 'cifar100_ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.01'  # hard, 0.5 test; hard for 1st order difference;; hard for 2nd order difference
    # experiment_dir = 'cifar100_ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.03'

    # experiment_dir = 'cifar100_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01'  # works only (???) with 2nd order difference; 2024-03-22 13:44:08,999 [INFO] Test accuracy/Train at epoch 40: 0.95655
    # experiment_dir = 'cifar10_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.03'

    # experiment_dir = 'cifar100_ResNet18_attacker=L2DeepFool_epsilon=0.01'  # hard, 0.5 test for 2nd order; check what's inside

    # experiment_dir = 'ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01'   #  0 ok
    experiment_dir = 'ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.01'  #

    # experiment_dir = 'cifar100_L2ProjectedGradientDescent_epsilon=0.01_L1ProjectedGradientDescent_epsilon=0.01'  # evaluate on two merged datasets

    assert experiment_dir in os.listdir(root_images_attack), (f'{experiment_dir} is not a subdirectory of '
                                                              f'{root_images_attack}')
    logging.info(f'{experiment_dir=}')

    train_path = f'{root_images_attack}/{experiment_dir}/train'
    test_path = f'{root_images_attack}/{experiment_dir}/test'

    assert os.path.isdir(train_path), f'{train_path} does not exist'
    assert os.path.isdir(test_path), f'{test_path} does not exist'

    logging.info(f'{train_path=}')
    logging.info(f'{test_path=}')

    epochs = 1000

    batch_sizes = {
        'racheta1': 1200,
        'racheta2': 1200,
        'racheta4': 160,
        'racheta10': 500,
        'ares': 160
    }
    batch_size = batch_sizes[machine_name]

    lr = 1e-3
    model_type = 'resnet50'

    logging.info(f'{epochs=} {batch_size= } {lr= } {model_type= } {difference=} ')

    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    dilation = None

    model = get_model(device=device, model_type=model_type, dilation=dilation)

    logging.info(f'Model name: {get_model_name(model)}')
    logging.info(model)
    logging.info(summary(model))

    experiment_name = (f'{experiment_dir}_{model_type}_'
                       f'{now_str}_epochs={epochs}_batch={batch_size}_lr={lr}_diff={difference}')

    if can_compile_torch_model(use_cuda=True, compile=False):
        model = torch.compile(model)
        logging.info('The model is compiled')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    logging.info(f'Optimizer: {get_optimizer_name(optimizer)}')

    # scheduler
    step_size = 10
    gamma = 0.9
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    logging.info(f'Scheduler: {get_scheduler_name(scheduler)}, {step_size=}, {gamma=}')
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4, verbose=True)

    experiment_name = (f'{experiment_name}_optimizer={get_optimizer_name(optimizer)}_'
                       f'scheduler={get_scheduler_name(scheduler)}')
    writer = SummaryWriter(comment=experiment_name)

    checkpoints_path = f'../saved_models/{experiment_name}'
    os.makedirs(checkpoints_path, exist_ok=True)

    train_model(model, device, train_path, test_path, epochs=epochs, batch_size=batch_size,
                optimizer=optimizer, scheduler=scheduler, writer=writer, log_interval=5, difference=difference,
                save_to=checkpoints_path)
    path_model = f'{checkpoints_path}/{experiment_name}.pth'
    torch.save(model.state_dict(), path_model)

    test_acc = test_model(model, device, test_path, batch_size, difference=difference)
    writer.add_scalar("Test", test_acc)
    writer.add_hparams(
        {
            "lr": lr, "batch_size": batch_size, "epochs": epochs, "model": model_type, "device": device,
            "optimizer": get_optimizer_name(optimizer),
        },
        {
            "test_accuracy": test_acc
        }
    )
    writer.close()

    # check if cifar100_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01 does not work with difference=1 - not working
    # check if cifar100_ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01 does work with difference=2


if __name__ == '__main__':
    caller_module: str = __file__.split(os.sep)[-1].split('.')[0]
    setup_logger(caller_module)
    main()
