from abc import ABC, abstractmethod
from itertools import count
import logging
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


logger = logging.getLogger('torchutil')


class ConcatDataset(Dataset):
    def __init__(self, *datasets: Dataset):
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)


class KLDivLogitLoss(nn.KLDivLoss):
    def __init__(self, *,
                 temperature: float = 1,
                 reduction='batchmean',
                 **kwargs):
        try:
            if not kwargs['log_target']:
                raise ValueError
        except KeyError:
            kwargs['log_target'] = True
        kwargs['reduction'] = reduction
        super().__init__(**kwargs)
        self.temperature = temperature

    # pylint: disable=redefined-builtin
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = F.log_softmax(input / self.temperature, dim=-1)
        target = F.log_softmax(target / self.temperature, dim=-1)
        return super().forward(input, target) * (self.temperature ** 2)


class Net(nn.Module, ABC):
    @abstractmethod
    def reset_parameters(self) -> None:
        pass


def l2_normalize(inputs: torch.Tensor, dim: int, epsilon: float = 1e-6):
    # for numerical stability, set the denominator to epsilon
    # if norm is less than epsilon
    return inputs.div(inputs.norm(p=2, dim=dim, keepdim=True).clip(epsilon))


def train_classifier(net: Net,
                     loader: DataLoader,
                     loss_fn: Callable[[Tensor, Tensor], Tensor],
                     optimizer: Optimizer,
                     device: torch.device,
                     log_every: int,
                     early_stopping: float,
                     epochs: int,
                     prefix: str = '',
                     leave: bool = True,
                     clip_gradient: float = 0.1,
                     ) -> None:
    if prefix:
        prefix += ' '           # add trailing space
    net.train()
    iterator = count() if early_stopping > 0 else range(epochs)
    for epoch in tqdm(iterator, leave=leave, desc='Training'):
        running_loss = 0.0
        correct = total = 0
        for index, data in enumerate(loader):
            inputs = data[0].to(device=device)
            labels = data[1].to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # clip gradients
            if clip_gradient > 0:
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(min=-clip_gradient,
                                               max=clip_gradient)

            optimizer.step()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                if index % log_every == log_every - 1:
                    logger.info('[%sEpoch %d Batch %5d] loss: %.3f',
                                prefix, epoch, index, running_loss / log_every)
                    running_loss = 0
        accuracy = correct / total
        logger.info('[Epoch %d] Training Accuracy: %.3f', epoch, accuracy)
        if early_stopping > 0:
            if accuracy > early_stopping:
                return
            if (epoch % 50 == 49) and (accuracy < 0.2):
                # not converging
                logger.warning('Network not converging. Reset Parameters.')
                net.reset_parameters()


def evaluate_classifier(net: nn.Module,
                        loader: DataLoader,
                        device: torch.device) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            images = data[0].to(device=device)
            labels = data[1].to(device=device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
