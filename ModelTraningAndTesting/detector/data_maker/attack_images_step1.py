import logging
import os
import shutil
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

import eagerpy as ep
import torch
from foolbox import PyTorchModel, Attack
from foolbox.attacks import L2DeepFoolAttack, L2ProjectedGradientDescentAttack, \
    L1ProjectedGradientDescentAttack, LinfFastGradientAttack, LinfProjectedGradientDescentAttack

from data_maker.dataset_manipulator import resize_split_data
from utils.environment_utils import is_windows_os
from utils.log_utils import setup_logger
from utils.stats import compute_mean_std
from utils.timing import timeit


@dataclass
class AttackDataSet:
    train_name: str
    test_name: str
    mean: torch.Tensor
    std: torch.Tensor
    train_loader: DataLoader
    test_loader: DataLoader


def get_dataset(split_path: str, dataset_name: str) -> AttackDataSet:
    batch_size = 50
    lst_transforms = [transforms.Resize((224, 224)), transforms.ToTensor()]
    transform = transforms.Compose(lst_transforms)
    dataset_name_train = f'{dataset_name}_train'
    split_path_train = os.path.join(split_path, 'train')

    dataset_name_test = f'{dataset_name}_test'
    split_path_test = os.path.join(split_path, 'val')

    trainset, testset = (ImageFolder(root=split_path_train, transform=transform),
                         ImageFolder(root=split_path_test, transform=transform))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                              num_workers=0 if is_windows_os() else 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0 if is_windows_os() else 2)

    mean, std = compute_mean_std(trainset)

    return AttackDataSet(train_name=dataset_name_train,
                         test_name=dataset_name_test,
                         mean=mean,
                         std=std,
                         train_loader=trainloader,
                         test_loader=testloader)


class ImageAttacker:
    def __init__(self, model: PyTorchModel,
                 model_name: str,
                 attacker: Attack,
                 attacker_name: str,
                 dataloader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 output_root_dir: str,
                 epsilon: float,
                 device: torch.device):
        self.model = model
        self.model_name = model_name
        self.attacker = attacker
        self.attacker_name = attacker_name
        self.epsilon = epsilon
        self.dataloader = dataloader
        self.dataset_name = dataset_name
        self.output_root_dir = output_root_dir
        self.device = device

    def _attack_details_(self) -> str:
        return f'{self.model_name}_attacker={self.attacker_name}_epsilon={self.epsilon}'

    def output_directory(self) -> str:
        return os.path.join(self.output_root_dir, self._attack_details_())

    def attack_and_write(self, extension='png'):
        directory = self.output_directory()
        print(f'Recreating directory {directory}')
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)
        index = 0
        for images, labels in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            images, labels = ep.astensors(*[images, labels])

            raw_advs, clipped_advs, success = self.attacker(self.model, images, labels, epsilons=[self.epsilon])
            for clipped_adv, label in zip(clipped_advs[0], labels):
                pil_image = transforms.ToPILImage()(torch.tensor(clipped_adv.numpy()))
                class_directory = os.path.join(directory, str(label.item()))
                os.makedirs(class_directory, exist_ok=True)
                full_file_path = os.path.join(class_directory, f'{self._attack_details_()}_{index}.{extension}')
                pil_image.save(full_file_path)
                index += 1


def get_attacker_name(attacker) -> str:
    attacker_name: str = attacker.__class__.__name__
    if attacker_name.endswith('Attack'):
        attacker_name = attacker_name[:-len('Attack')]
    return attacker_name


@timeit
def main():
    caller_module: str = __file__.split(os.sep)[-1].split('.')[0]
    setup_logger(caller_module)

    new_size: Tuple[int, int] = (224, 224)  # put it in config file
    min_images: int = 40

    input_path: str = '../data/german/archive/train'
    resized_path: str = f'../data/german/DATA_{new_size[0]}x{new_size[1]}'
    split_path: str = '../data/german/split'

    output_path, kept_directories = resize_split_data(input_path=input_path,
                                                      resized_path=resized_path,
                                                      output_path=split_path,
                                                      new_size=new_size,
                                                      min_images=min_images,
                                                      ratio=(0.8, 0.2),
                                                      seed=42,
                                                      move=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'{device=}')

    attacks = [
        LinfProjectedGradientDescentAttack(),
               LinfFastGradientAttack(),  # FGSM
               L1ProjectedGradientDescentAttack(),  # L1PGD
               L2ProjectedGradientDescentAttack(),  # L2PGD
               # L2DeepFoolAttack(loss='crossentropy')
               ]

    data = get_dataset(split_path, 'gtsrb')

    attack_directory = f'../data/german/_attacked'
    if os.path.exists(attack_directory):
        shutil.rmtree(attack_directory, ignore_errors=True)

    for attack in tqdm(attacks, desc='Attacks', initial=1):
        epsilons = [
            #first stage
            # 0.0002,
            # 0.0005,
            # 0.0008,
            # 0.001,
            # 0.0015,

            # second stage
            # 0.002,
            # 0.003,
            0.01,
            # 0.03,
            # 0.1,
            # 0.3,

            # third stage
            # 0.5,
            # 1.0,
        ]

        for loader, dataset_name in zip((data.train_loader, data.test_loader), (data.train_name, data.test_name)):
            for epsilon in epsilons:
                model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
                preprocessing = {'mean': data.mean.to(device), 'std': data.std.to(device), 'axis': -3}
                fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
                attack_name = get_attacker_name(attack)
                attacker = ImageAttacker(fmodel, 'ResNet18', attack, attack_name, loader, dataset_name,
                                         f'{attack_directory}/{dataset_name}', epsilon=epsilon,
                                         device=device)
                attacker.attack_and_write()


if __name__ == '__main__':
    main()

    # FGM = L2FastGradientAttack
    # FGSM = LinfFastGradientAttack
    # L1PGD = L1ProjectedGradientDescentAttack
    # L2PGD = L2ProjectedGradientDescentAttack
    # LinfPGD = LinfProjectedGradientDescentAttack
    # PGD = LinfPGD
    # MIFGSM = LinfMomentumIterativeFastGradientMethod
    #
    # L1AdamPGD = L1AdamProjectedGradientDescentAttack
    # L2AdamPGD = L2AdamProjectedGradientDescentAttack
    # LinfAdamPGD = LinfAdamProjectedGradientDescentAttack
    # AdamPGD = LinfAdamPGD
