import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class DiffDataset(Dataset):
    def __init__(self, original_dataset: Dataset, difference: int = 1, dim=1) -> None:
        assert difference >= 1, f'Should be differentiation of at least degree 1, got {difference=}'
        super().__init__()
        self.original_dataset = original_dataset
        self.difference = difference
        self.dim = dim

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index):
        item, class_index = self.original_dataset[index]
        item = torch.diff(item, n=self.difference, dim=self.dim)
        return item, class_index
