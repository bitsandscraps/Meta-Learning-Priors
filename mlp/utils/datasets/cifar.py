import random
from os.path import join
import pickle
from typing import Callable, Optional, Tuple, Type, Union

import numpy.random as npr
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from .misc import ROOT


STAT = Tuple[float, float, float]


def get_stats(dataset: Union[Type[CIFAR10], Type[CIFAR100]],
              name: str) -> Tuple[STAT, STAT]:
    path = join(ROOT, name + '.pkl')
    try:
        with open(path, 'rb') as stats:
            return pickle.load(stats)
    except FileNotFoundError:
        dataset_ = dataset(ROOT, transform=transforms.ToTensor(),
                           train=True, download=True)
        loader = DataLoader(dataset_, batch_size=len(dataset_), shuffle=False)
        batch = next(iter(loader))[0]
        means = (batch[:, 0, ...].mean().item(),
                 batch[:, 1, ...].mean().item(),
                 batch[:, 2, ...].mean().item())
        stds = (batch[:, 0, ...].std().item(),
                batch[:, 1, ...].std().item(),
                batch[:, 2, ...].std().item())
        with open(path, 'wb') as stats:
            pickle.dump((means, stds), stats)
        return means, stds


def get_train_transform(means: STAT, stds: STAT):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])


def get_test_transform(means: STAT, stds: STAT):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])


def get_CIFAR10(valid_ratio: float,
                train_transform=get_train_transform,
                test_transform=get_test_transform,
                ) -> Tuple[Tuple[Dataset, Dataset, Dataset], torch.Size, int]:
    stats = get_stats(CIFAR10, 'cifar10')
    train_transform_ = train_transform(*stats)
    test_transform_ = test_transform(*stats)
    train = CIFAR10(ROOT, train=True, transform=train_transform_, download=True)
    test = CIFAR10(ROOT, train=False, transform=test_transform_, download=True)
    test_size = len(test)
    valid_size = round(test_size * valid_ratio)
    test_size -= valid_size
    valid, test = random_split(test, (valid_size, test_size))
    return (train, valid, test), train[0][0].size(), 10


def get_CIFAR10_v2(valid_ratio: float,
                   train_transform=get_train_transform,
                   test_transform=get_test_transform,
                   seed: Optional[int] = None
                   ) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    stats = get_stats(CIFAR10, 'cifar10')
    train_transform_ = train_transform(*stats)
    test_transform_ = test_transform(*stats)
    train = CIFAR10(ROOT, train=True, transform=train_transform_, download=True)
    train_test = CIFAR10(ROOT, train=True,
                         transform=test_transform_, download=True)
    test = CIFAR10(ROOT, train=False, transform=test_transform_, download=True)
    train_size = len(train)
    indices = list(range(train_size))
    if seed is None:
        random.shuffle(indices)
    else:
        rng = npr.default_rng(seed)
        rng.shuffle(indices)
    valid_size = round(train_size * valid_ratio)
    train_indices = list(indices[valid_size:])
    valid_indices = list(indices[:valid_size])
    train = Subset(train, train_indices)
    valid = Subset(train_test, valid_indices)
    train_test = Subset(train_test, train_indices)
    return train, train_test, valid, test


def get_CIFAR100(valid_ratio: float,
                 train_transform=get_train_transform,
                 test_transform=get_test_transform,
                 ) -> Tuple[Tuple[Dataset, Dataset, Dataset], torch.Size, int]:
    stats = get_stats(CIFAR100, 'cifar100')
    train_transform_ = train_transform(*stats)
    test_transform_ = test_transform(*stats)
    train = CIFAR100(ROOT, train=True,
                     transform=train_transform_, download=True)
    train_size = len(train)
    valid_size = round(train_size * valid_ratio)
    train_size -= valid_size
    train, valid = random_split(train, (train_size, valid_size))
    test = CIFAR100(ROOT, train=False, transform=test_transform_, download=True)
    return (train, valid, test), train[0][0].size(), 100
