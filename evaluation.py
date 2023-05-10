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

def plot_tensor(subplot: int, tensor: torch.Tensor) -> None:
    assert tensor.shape == (28, 28)
    plt.subplot(2, 5, subplot)
    plt.matshow(
        np.array(
            tensor,
            dtype=np.float32,
        ),
        fignum=False,
    )


def plot_tensors(tensors: torch.Tensor) -> None:
    tensors = tensors.cpu()
    for subplot, tensor in enumerate(tensors):
        plot_tensor(subplot+1, tensor.reshape(28, 28))
        

# def optimize_n(
#     device: torch.device,
#     model: Net,
#     n: int
# ) -> torch.Tensor:
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
#     ])
#     def get_rand_data() -> torch.Tensor:
#         # return transform(
#         #     torch.randint(0, 255, (1, 28, 28), dtype=torch.float32)
#         # )
#         rand_uint8: np.ndarray = np.random.randint(0, 255, (1, 28, 28), dtype=np.uint8)
#         # rand_uint8.resize((1, 28, 28))
#         transformed: torch.Tensor = transform(rand_uint8)
#         transformed.resize((1, 28, 28))
#         return transformed
        
        
    
#     model.to('cpu')
    
#     data = torch.nn.Parameter(
#         get_rand_data(),
#         requires_grad=True
#     )
#     # data.to(device)
#     # print(data)
#     # raise ValueError
#     model.requires_grad_(False)
#     optimizer = torch.optim.SGD(
#         [data],
#         lr = .1,
#         # weight_decay=.99,
#     )
#     # mse = torch.nn.MSELoss()
#     # target = torch.zeros(1, 10)
#     # target[0][n] = 1
#     # target = torch.zeros(10)
#     # target[n] = 1
#     target = torch.tensor([n])
    
#     for epoch in range(10000):
#         output = model(data)
#         # loss = mse(output, target)
#         loss = nn.functional.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         if epoch % 1000 == 0:
#             print(f'loss: {loss}')

#     return data
    



def load_and_do_stuff(path: str) -> None:
    device = get_device()

    model = Net().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    # print('state_dict:')
    # for param_tensor in model.state_dict():
    #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    
    print(f'\nmodel:\n{model}\n')
    
    plot_tensors(model.state_dict()['layer1.weight'])
    plt.show()
    
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
    load_and_do_stuff('models/total_epochs_32_gamma_70_random_seed_1_weight_decay_90_noise_std_100_eval_86.pth')


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
