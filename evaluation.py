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
import matplotlib.pyplot as plt # type: ignore
from my_utils import *

def draw_layer_for_target(model: Net, target_n: int) -> None:
    weights = np.array(
        model.state_dict()['layer1.weight'][target_n].cpu(),
        dtype=np.float32
    ).reshape(28, 28)
    # print(weights)
    # print(weights.shape)
    plt.matshow(weights)
    plt.show()

def load_and_do_stuff(path: str) -> None:
    device = get_device()

    model = Net().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    # print('state_dict:')
    # for param_tensor in model.state_dict():
    #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    
    print(f'\nmodel:\n{model}\n')
    
    for target_n in range(10):
        draw_layer_for_target(model, target_n)
    
    # def rand_input() -> torch.Tensor:
    #     raise NotImplemented
    #     # return transform(np.random.random((1, 28, 28)).astype(np.float32))
    #     # return torch.rand([1, 1, 28, 28], device=device)
    
    # target_n: Final[int] = 0
    # target_tensor: Final[torch.Tensor] = torch.Tensor([target_n]).to(device)
    # # target_tensor: torch.Tensor = torch.zeros([10], device=device)
    # # target_tensor[target_n] = 1
    # for _ in range(2):
    #     image = rand_input()
    #     # print(image)
    #     # raise ValueError
    #     output = model(image)
    #     loss = nn.functional.nll_loss(output, target_tensor, reduction='sum').item()
    #     # loss = nn.functional.nll_loss(output, target_tensor)
    #     print(loss)



if __name__ == '__main__':
    raise NotImplemented


# old file names
# lots of things (with noise bc i forgot)
# load_and_do_stuff('alphanumeric/noise_leaky_relu_single_layer_92.pth') # 3 looks nice
# load_and_do_stuff('alphanumeric/noise_relu_single_layer_84.pth') # 3 failed lmao
# load_and_do_stuff('alphanumeric/noise_single_layer_adam_89.pth') # looks noisy
# load_and_do_stuff('alphanumeric/noise_single_layer_l2_decay_87.pth') # LOOKS INSANELY GOOD

# l2_NN, Adadelta, single layer (28*28->10), (noise_0)
# load_and_do_stuff('alphanumeric/single_layer_l2_01_92.pth')
# load_and_do_stuff('alphanumeric/single_layer_l2_10_90.pth')
# load_and_do_stuff('alphanumeric/single_layer_l2_90_87.pth')
# load_and_do_stuff('alphanumeric/single_layer_l2_99_86.pth')

# noise_NN, l2_10, Adadelta, single layer (28*28->10)
# they kinda get more spread out and not weigh the entire curve equally
# load_and_do_stuff('alphanumeric/single_layer_noise_01_l2_10_90.pth')
# load_and_do_stuff('alphanumeric/single_layer_noise_10_l2_10_90.pth')
# load_and_do_stuff('alphanumeric/single_layer_noise_50_l2_10_90.pth') # too much noise

# noise_NN, l2_90, Adadelta, single layer (28*28->10)
# load_and_do_stuff('alphanumeric/single_layer_noise_01_l2_90_87.pth')
# load_and_do_stuff('alphanumeric/single_layer_noise_10_l2_90_87.pth')
# load_and_do_stuff('alphanumeric/single_layer_noise_50_l2_90_87.pth') # starts looking kinda noisy
# load_and_do_stuff('alphanumeric/single_layer_noise_100_l2_90_85.pth')

# 10 epochs, leaky_relu
# load_and_do_stuff('alphanumeric/leaky_relu_single_layer_noise_100_l2_90_85.pth') # noisy bc 10 epochs isn't enough, but doesn't look much better than without leaky_relu

# more epochs, noise_100, l2_90, Adadelta, single layer (28*28->10)
# load_and_do_stuff('alphanumeric/single_layer_noise_100_l2_90_epochs_20_86.pth') # slightly noisy
# load_and_do_stuff('alphanumeric/single_layer_noise_100_l2_90_epochs_32_86.pth') # no noise
    