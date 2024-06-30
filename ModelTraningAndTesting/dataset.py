import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class AdeversarialDataset(Dataset):
    def __init__(self, dataset_path: str, transform: transforms.Compose = None) -> None:
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.labels = []
        self.images = []

        for path in Path(dataset_path).joinpath("0=genuine").glob("*.png"):
            self.labels.append(0)
            image = Image.open(path)
            image_tensor = self.transform(image)
            self.images.append(image_tensor)


        for i, path in enumerate(Path(dataset_path).joinpath("1=attacked").glob("*.png")):
            if i >= 7842:
                break
            self.labels.append(1)
            image = Image.open(path)
            image_tensor = self.transform(image)
            self.images.append(image_tensor)
        
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index) -> tuple[torch.tensor, int]:
        return self.images[index], self.labels[index]
        

dataset = AdeversarialDataset("./data/german/detect_attack/ResNet18_attacker=L2DeepFool_epsilon=0.01/test", transform=transform)
