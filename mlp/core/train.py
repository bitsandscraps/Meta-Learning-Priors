from argparse import ArgumentParser
from functools import partial
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange

from ..utils import add_defaults, set_seed
from ..utils.datasets.mnist import get_MNIST
from ..utils.logger import Logger
from .module import EnergyEncoder, EnergyEncoderDeepSet, LeNet, Net
from .mcmc import HMC, sample_prior, sample_posterior


ANGLES = (0, 15, 30, 45, 60, 75)


class RotationTransform(nn.Module):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def forward(self, image):
        return TF.rotate(image, self.angle)


def transform(angle, mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        RotationTransform(angle),
        transforms.Normalize((mean,), (std,))])


def rotated_mnist():
    trainsets = {}
    testsets = {}
    for angle in ANGLES:
        (train, _, test), shape, _ = get_MNIST(0,
                                               partial(transform, angle),
                                               partial(transform, angle))
        trainsets[angle] = train
        testsets[angle] = test
    return trainsets, testsets, shape


def evaluate(device: torch.device,
             energy_encoder: EnergyEncoder,
             leapfrog_steps: int,
             trainloader: Optional[DataLoader],
             testloader: DataLoader,
             priors: List[Tuple[Net, Net, HMC]],
             posteriors: List[Tuple[Net, Net, HMC]],
             steps: int) -> float:
    if trainloader is None:
        for model, target, optim in tqdm(priors, leave=False, desc='evaluate'):
            sample_prior(model, target, optim, energy_encoder, leapfrog_steps, steps)
    else:
        train_images, train_labels = get_batches(device, trainloader)
        for model, target, hmc in tqdm(posteriors, leave=False, desc='posterior'):
            sample_posterior(model, target, hmc, energy_encoder,
                             train_images, train_labels, leapfrog_steps, steps)
    total = correct = 0
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(device=device)
            labels = data[1].to(device=device)
            outputs = []
            for model, _, _ in priors:
                outputs.append(F.softmax(model(images), dim=1))
            output = torch.stack(outputs).mean(0)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def get_likelihoods(images: torch.Tensor,
                    labels: torch.Tensor,
                    model: nn.Module) -> float:
    log_probs = Categorical(logits=model(images)).log_prob(labels)
    return np.exp(log_probs.sum().item())


def get_average_prob(images: torch.Tensor,
                     labels: torch.Tensor,
                     distributions: List[Tuple[Net, Net, HMC]]):
    outputs = []
    for model, _, _ in distributions:
        outputs.append(get_likelihoods(images, labels, model))
    return np.mean(outputs)


def get_batches(device: torch.device, loader: DataLoader):
    data = next(iter(loader))
    images = data[0].to(device=device)
    labels = data[1].to(device=device)
    return images, labels


def train(device: torch.device,
          energy_encoder: EnergyEncoder,
          hmc_steps: int,
          trainloaders: List[DataLoader],
          testloaders: List[DataLoader],
          leapfrog_steps: int,
          optimizer: Adam,
          priors: List[Tuple[Net, Net, HMC]],
          posteriors: List[Tuple[Net, Net, HMC]]):
    trainloader = random.choice(trainloaders)
    testloader = random.choice(testloaders)
    train_images, train_labels = get_batches(device, trainloader)
    test_images, test_labels = get_batches(device, testloader)
    # union train and test
    both_images = torch.cat([train_images, test_images], dim=0)
    both_labels = torch.cat([train_labels, test_labels], dim=0)

    for model, target, hmc in tqdm(priors, leave=False, desc='prior'):
        sample_prior(model, target, hmc, energy_encoder,
                     leapfrog_steps, hmc_steps)
        model.freeze()

    for model, target, hmc in tqdm(posteriors, leave=False, desc='posterior'):
        sample_posterior(model, target, hmc, energy_encoder,
                         train_images, train_labels, leapfrog_steps, hmc_steps)
        model.freeze()

    optimizer.zero_grad()
    with torch.no_grad():
        testlls = np.asarray([get_likelihoods(test_images, test_labels, model)
                              for model, _, _ in posteriors])
        numerator = get_average_prob(both_images, both_labels, priors)
        denominator = get_average_prob(train_images, train_labels, priors)
        if np.isclose(numerator, denominator):
            ratio = 1
        else:
            ratio = numerator / denominator
        weights = testlls - ratio

    pri_energy_sum = 0.
    for model, _, _ in priors:
        pri_energy_sum += energy_encoder(model)
    pri_energy_mean = pri_energy_sum / len(priors)

    loss = 0.
    for weight, (model, _, _) in zip(weights, posteriors):
        loss += weight * (energy_encoder(model) - pri_energy_mean)
    loss = loss / len(posteriors)
    loss.backward()
    optimizer.step()

    for model, _, _ in priors + posteriors:
        model.unfreeze()


def get_triples(device: torch.device,
                mass: float,
                samples: int,
                step_size: float):
    triples = []
    for _ in range(samples):
        model = LeNet(10).to(device=device)
        triples.append((model,
                        LeNet(10).to(device=device),
                        HMC(model.parameters(), step_size, mass)))
    return triples


def main(aggregator: str,
         device: torch.device,
         hmc_steps: int,
         posterior_samples: int,
         prior_samples: int,
         learning_rate: float,
         leapfrog_steps: int,
         logger: Logger,
         mass: float,
         seed: int,
         shots: int,
         steps: int,
         step_size: float,
         target_domain: int,
         test_batch_size: int,
         train_batch_size: int,
         ):
    set_seed(seed)
    trainsets, testsets, shape = rotated_mnist()
    testset = testsets.pop(target_domain)
    trainset = trainsets.pop(target_domain)
    if shots > 0:
        trainloader = DataLoader(trainset, shots * 10, shuffle=True)
    else:
        trainloader = None
    testloader = DataLoader(testset, 1000, shuffle=False)
    domains = list(testsets.keys())
    random.shuffle(domains)
    metatrainloaders = [DataLoader(trainsets[key], batch_size=train_batch_size, shuffle=True)
                        for key in domains[:3]]
    metatestloaders = [DataLoader(trainsets[key], batch_size=test_batch_size, shuffle=True)
                       for key in domains[3:]]
    priors = get_triples(device, mass, prior_samples, step_size)
    posteriors = get_triples(device, mass, posterior_samples, step_size)
    if aggregator == 'transform':
        energy_encoder = EnergyEncoder(32, shape, 4, 32, 10).to(device=device)
    elif aggregator == 'deepset':
        energy_encoder = EnergyEncoderDeepSet(32, shape, 32, 10).to(device=device)
    else:
        raise NotImplementedError(aggregator)
    optimizer = Adam(energy_encoder.parameters(), lr=learning_rate)
    for _ in trange(steps):
        train(device, energy_encoder, hmc_steps, metatrainloaders,
              metatestloaders, leapfrog_steps, optimizer, priors, posteriors)
        logging.info(evaluate(device, energy_encoder, leapfrog_steps,
                              trainloader, testloader, priors, posteriors, hmc_steps))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--aggregator', choices=['transform', 'deepset'],
                        default='transform')
    parser.add_argument('--hmc-steps', type=int, default=30)
    parser.add_argument('--leapfrog-steps', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--mass', type=float, default=1)
    parser.add_argument('--posterior-samples', type=int, default=32)
    parser.add_argument('--prior-samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--step-size', type=float, default=0.003)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--target-domain', choices=ANGLES, default=75)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--train-batch-size', type=int, default=64)
    main(**add_defaults(parser, 'mlp'))
