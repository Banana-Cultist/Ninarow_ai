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

class AddGaussianNoise(object):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0
    ):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def train_epoch(
    log_interval: float,
    model: Net,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader[datasets.MNIST],
    optimizer: torch.optim.Optimizer,
) -> None:
    assert train_loader.batch_size is not None
    # batch_count = len(train_loader)
    # batch_size = train_loader.batch_size
    # approx_dataset_size = batch_count*batch_size
    dataset_size = len(train_loader.dataset) # type: ignore
    model.train()
    epoch_start = timeit.default_timer()
    start = timeit.default_timer()
    for batch_index, (data, target) in enumerate(train_loader):
        # print(data[0])
        # raise ValueError
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        # loss = nn.functional.mse_loss(output, target)
        # print('output:', output[0])
        # print('target:', target[0])
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        
        if timeit.default_timer() - start > log_interval:
        # if batch_index % log_interval == 0:
            start = timeit.default_timer()
            numerator = batch_index * len(data)
            denominator = dataset_size
            print(
                f'batch {batch_index} ({numerator}/{denominator}) ({(100 * numerator / denominator):.0f}%)\tLoss: {loss.item():.6f}'
            )
    epoch_end = timeit.default_timer()
    delta: Final[float] = epoch_end-epoch_start
    print()
    print(f'time for epoch: {delta}')
    print(f'time per batch: {delta / train_loader.batch_size}')
    print(f'time per item: {delta / dataset_size}')


def test(
    model: Net,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader[datasets.MNIST]
) -> float:
    model.eval()
    test_loss: float = 0
    correct: int = 0
    dataset_size = len(test_loader.dataset) # type: ignore
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= dataset_size
    print(
        f'Test set: loss: {test_loss:.4f}, accuracy: {correct}/{dataset_size} ({100 * correct / dataset_size:.1f}%)\n'
    )
    return correct / dataset_size

def get_data(noise_std: float) -> Tuple[Any, Any]:
    transform = transforms.Compose([
    # transform = torch.nn.Sequential([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0, noise_std),
    ])
    # transform = torch.jit.script(transform)

    train_dataset = datasets.MNIST(
        'data',
        train=True,
        transform=transform,
        download=False,
    )
    test_dataset = datasets.MNIST(
        'data',
        train=False,
        transform=transform,
        download=False,
    )
    return train_dataset, test_dataset
    

def get_dataloaders(
    device: torch.device,
    train_batch_size: int,
    test_batch_size: int,
    noise_std: float,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader
]:
    
    train_dataset, test_dataset = get_data(noise_std)
    
    cuda_kwargs: Final[dict[str, Any]] = {
        'num_workers': 1,
        'pin_memory': True,
    } if device.type == 'cuda' else {}
    
    train_dataloader: torch.utils.data.DataLoader[datasets.MNIST] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        shuffle = True,
        **cuda_kwargs
    )
    
    
    test_dataloader: torch.utils.data.DataLoader[datasets.MNIST] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = test_batch_size,
        shuffle = False,
        **cuda_kwargs
    )
    
    return train_dataloader, test_dataloader

def train_and_save(
    total_epochs: int = 10,
    learning_rate_decay: float = 0.7,
    random_seed: Optional[int] = 1,
    weight_decay: float = .9,
    noise_std: float = 1.0
) -> None:
    train_batch_size: int = 64
    test_batch_size: int = 1000
    learning_rate: float = 1.0
    learning_rate_decay_epochs: int = 1
    log_interval: float = 1.0 # how many seconds to wait before logging training status

    model_name: Final[str] = f'''
    total_epochs_{total_epochs}_
    gamma_{int(100*learning_rate_decay)}_
    random_seed_{random_seed}_
    weight_decay_{int(100*weight_decay)}_
    noise_std_{int(noise_std)}
    '''.replace('\n', '').replace('\t', '').replace(' ', '')
    print(f'training {model_name}')
    # raise ValueError

    if random_seed is not None:
        torch.manual_seed(random_seed)

    device = get_device()
    
    train_dataloader, test_dataloader =  get_dataloaders(
        device,
        train_batch_size,
        test_batch_size,
        noise_std,
    )

    model = Net().to(device)
    # model = torch.nn.DataParallel(model) # not faster
    # model = torch.nn.DistributedDataParallel(model) # not a thing
    # model = torch.compile(model) # unsupported for python 3.11
    
    optimizer: torch.optim.Optimizer = torch.optim.Adadelta(
    # optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = learning_rate_decay_epochs,
        gamma = learning_rate_decay,
    )
    
    # example_input: torch.Tensor = train_dataloader.dataset.test_data[0]
    # example_input: torch.Tensor = torch.rand([train_batch_size, 1, 28, 28], dtype=torch.float32)
    # example_input = example_input.to(device)
    # traced_model = torch.jit.script(
    #     model,
    #     # example_input,
    # )
    
    start = timeit.default_timer()
    test_correct: Optional[float] = None
    for cur_epoch in range(total_epochs):
        print(f'\ntraining epoch {cur_epoch+1}')
        train_epoch(
            log_interval,
            model,
            device,
            train_dataloader,
            optimizer,
        )
        test_correct = test(model, device, test_dataloader)
        scheduler.step()
        # raise ValueError
    end = timeit.default_timer()
    print(f'time for training: {end-start}')
    assert test_correct is not None
    
    # torch.save(
    #     model.state_dict(),
    #     f'alphanumeric/single_layer_noise_100_l2_90_epochs_32{str(datetime.datetime.now(datetime.timezone.utc))}.pth'
    # )
    
    torch.save(
        model.state_dict(),
        f'models/{model_name}_eval_{int(100*test_correct)}.pth'
    )

if __name__ == '__main__':
    train_and_save(
        
    )