from os.path import join
import pickle
from typing import Callable, Optional, Tuple, Type

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from .misc import ROOT


def get_stats(dataset: Type[MNIST],
              name: str) -> Tuple[float, float]:
    path = join(ROOT, name + '.pkl')
    try:
        with open(path, 'rb') as stats:
            return pickle.load(stats)
    except FileNotFoundError:
        dataset_ = dataset(ROOT, transform=transforms.ToTensor(),
                           train=True, download=True)
        loader = DataLoader(dataset_, batch_size=len(dataset_), shuffle=False)
        batch = next(iter(loader))[0]
        mean = batch[:, 0, ...].mean().item()
        std = batch[:, 0, ...].std().item()
        with open(path, 'wb') as stats:
            pickle.dump((mean, std), stats)
        return mean, std


def get_transform(mean: float, std: float):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))])


def get_MNIST(valid_ratio: float,
              train_transform=get_transform,
              test_transform=get_transform,
              ) -> Tuple[Tuple[Dataset, Dataset, Dataset], torch.Size, int]:
    stats = get_stats(MNIST, 'mnist')
    train_transform_ = train_transform(*stats)
    test_transform_ = test_transform(*stats)
    train = MNIST(ROOT, train=True, transform=train_transform_, download=True)
    test = MNIST(ROOT, train=False, transform=test_transform_, download=True)
    test_size = len(test)
    valid_size = round(test_size * valid_ratio)
    test_size -= valid_size
    valid, test = random_split(test, (valid_size, test_size))
    return (train, valid, test), train[0][0].size(), 10
