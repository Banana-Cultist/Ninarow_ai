# so imma do MNIST and then optimize for what it thinks is the most of any number
from __future__ import annotations
import timeit
from typing import * # type: ignore
import datetime
import numpy as np
import torch
import torch.backends.mps
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms # type: ignore
import torch.jit
import matplotlib.pyplot as plt # type: ignore

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Net1(nn.Module):
    # the weights that came with the tutorial
    def __init__(self) -> None:
        super(Net1, self).__init__()
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

class Net2(nn.Module):
    # other network
    def __init__(self) -> None:
        super(Net2, self).__init__()
        # self.layer1 = nn.Linear(28*28, 128)
        # self.layer2 = nn.Linear(128, 10)
        self.layer1 = nn.Linear(28*28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = nn.functional.leaky_relu(x, .2)
        # x = nn.functional.relu(x)
        # x = self.layer2(x)
        output = nn.functional.log_softmax(x, dim=1)
        # output = nn.functional.sigmoid(x)
        return output

Net = Net2

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
) -> None:
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

def get_data() -> Tuple[Any, Any]:
    transform = transforms.Compose([
    # transform = torch.nn.Sequential([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0, 1.00),
    ])
    # transform = torch.jit.script(transform)

    train_dataset = datasets.MNIST(
        'alphanumeric/data',
        train=True,
        transform=transform,
        download=False,
    )
    test_dataset = datasets.MNIST(
        'alphanumeric/data',
        train=False,
        transform=transform,
        download=False,
    )
    return train_dataset, test_dataset
    

def get_dataloaders(
    device: torch.device,
    train_batch_size: int,
    test_batch_size: int,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader
]:
    
    train_dataset, test_dataset = get_data()
    
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

def train_and_save() -> None:
    train_batch_size: Final[int] = 64
    test_batch_size: Final[int] = 1000
    
    total_epochs: Final[int] = 10
    learning_rate: Final[float] = 1.0
    learning_rate_decay_epochs: Final[int] = 1
    learning_rate_decay: Final[float] = 0.7

    random_seed: Final[Optional[int]] = 1
    if random_seed is not None:
        torch.manual_seed(random_seed)

    log_interval: Final[float] = 1.0 # how many seconds to wait before logging training status
    
    device = get_device()
    
    train_dataloader, test_dataloader =  get_dataloaders(
        device,
        train_batch_size,
        test_batch_size,
    )

    model = Net().to(device)
    # model = torch.nn.DataParallel(model) # not faster
    # model = torch.nn.DistributedDataParallel(model) # not a thing
    # model = torch.compile(model) # unsupported for python 3.11
    
    optimizer: torch.optim.Optimizer = torch.optim.Adadelta(
    # optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate,
        weight_decay=.90,
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
    for cur_epoch in range(total_epochs):
        print(f'\ntraining epoch {cur_epoch+1}')
        train_epoch(
            log_interval,
            model,
            device,
            train_dataloader,
            optimizer,
        )
        test(model, device, test_dataloader)
        scheduler.step()
        # raise ValueError
    end = timeit.default_timer()
    print(f'time for training: {end-start}')

    torch.save(
        model.state_dict(),
        f'alphanumeric/leaky_relu_single_layer_noise_100_l2_90_{str(datetime.datetime.now(datetime.timezone.utc))}.pth'
    )

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
    # get_data()
    # train_and_save()
    
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
    load_and_do_stuff('alphanumeric/single_layer_noise_100_l2_90_epoch_20_86.pth') # looks pretty good, only a little noisy
    
    