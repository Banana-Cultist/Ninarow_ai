from __future__ import annotations
import timeit
from typing import * # type: ignore
import datetime
import numpy as np
import torch
import torch.backends.mps
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms # type: ignore
import torch.jit
# import matplotlib.pyplot as plt # type: ignore



class _Net1(nn.Module):
    # the weights that came with the tutorial
    def __init__(self) -> None:
        super(_Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input has shape [batch_size, channels=1 (greyscale), height=28, width=28]
        # print(x.size())
        # raise ValueError
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, .2)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x, .2)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x, .2)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

class _Net2(nn.Module):
    # other network
    def __init__(self) -> None:
        super(_Net2, self).__init__()
        # self.layer1 = nn.Linear(28*28, 128)
        # self.layer2 = nn.Linear(128, 10)
        self.layer1 = nn.Linear(28*28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        # x = nn.functional.leaky_relu(x, .2)
        # x = nn.functional.relu(x)
        # x = self.layer2(x)
        output = nn.functional.log_softmax(x, dim=1)
        # output = nn.functional.sigmoid(x)
        return output

Net = _Net2

def get_device() -> torch.device:
    cuda_ok: Final[bool] = True    
    mps_ok: Final[bool] = True

    if cuda_ok and torch.cuda.is_available():
        print('using CUDA')
        return torch.device('cuda')
    elif mps_ok and torch.backends.mps.is_available():
        print('using macOS GPU')
        return torch.device('mps')
    else:
        print('using CPU')
        return torch.device('cpu')