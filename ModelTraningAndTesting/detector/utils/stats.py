from typing import Tuple

import torch
from datasets import Dataset

# https://www.youtube.com/watch?v=y6IEcEBRZks
def compute_mean_std(train_data: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    channel_sum, channel_squared_sum = 0, 0
    for data, _ in train_data:
        channel_sum += torch.mean(data, dim=[1, 2])
        channel_squared_sum += torch.mean(data**2, dim=[1, 2])

    mean = channel_sum / len(train_data)
    std = (channel_squared_sum / len(train_data) - mean**2) ** 0.5

    return mean, std

