import math
import random
from typing import Callable, List

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.optim import Optimizer

from .module import Net


class HMC(Optimizer):
    def __init__(self, params, step_size, mass):
        if mass < 0:
            raise ValueError(f'Invalid mass: {mass}')
        super().__init__(params, {'step_size': step_size, 'mass': mass})

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            mass = group['mass']
            for p in group['params']:
                param_state = self.state[p]
                # sample p0
                momentum = torch.randn_like(p).mul_(math.sqrt(mass))
                # store it
                param_state['momentum'] = momentum
                param_state['reset'] = True

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            step_size = group['step_size']
            mass = group['mass']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                param_state = self.state[p]
                if param_state['reset']:
                    momentum = param_state['momentum']
                    # update p0.5
                    momentum.sub_(d_p, alpha=step_size / 2)
                    # update reset
                    param_state['reset'] = False
                else:
                    # get previous half-step momentum
                    momentum = param_state['momentum']
                    # update it to next half-step momentum
                    momentum.sub_(d_p, alpha=step_size)
                p.add_(momentum, alpha=step_size / mass)

    @torch.no_grad()
    def step_final(self):
        for group in self.param_groups:
            step_size = group['step_size']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                param_state = self.state[p]
                param_state['momentum'].sub_(d_p, alpha=step_size / 2)

    @torch.no_grad()
    def kinetic_energy(self):
        energy = 0.
        for group in self.param_groups:
            mass = group['mass']
            group_energy = 0.
            for p in group['params']:
                param_state = self.state[p]
                if 'momentum' not in param_state:
                    raise ValueError('Leapfrog step is in progress.')
                group_energy += param_state['momentum'].square().sum().item()
            energy += group_energy / (2 * mass)
        return energy


def sample(model: Net,
           target: Net,
           optimizer: HMC,
           energy_function: Callable[[nn.Module], torch.Tensor],
           leapfrog_steps: int):
    model.reset_parameters()
    optimizer.zero_grad()
    target.load_state_dict(model.state_dict())
    optimizer.reset()
    energy = energy_function(model)
    initial_hamiltonian = energy.item() + optimizer.kinetic_energy()
    for _ in range(leapfrog_steps - 1):
        energy.backward()
        optimizer.step()
        optimizer.zero_grad()
        energy = energy_function(model)
    energy.backward()
    optimizer.step_final()
    final_hamiltonian = energy.item() + optimizer.kinetic_energy()
    acceptance_probability = math.exp(initial_hamiltonian - final_hamiltonian)
    if acceptance_probability < 1:
        # reject the sample
        if random.random() > acceptance_probability:
            model.load_state_dict(target.state_dict())


def sample_prior(model: Net,
                 target: Net,
                 optimizer: HMC,
                 energy_encoder: Net,
                 leapfrog_steps: int,
                 steps: int):
    energy_encoder.freeze()

    def energy_function(model_):
        return energy_encoder(model_)

    for _ in range(steps):
        sample(model, target, optimizer, energy_function, leapfrog_steps)

    energy_encoder.unfreeze()


def sample_posterior(model: Net,
                     target: Net,
                     optimizer: HMC,
                     energy_encoder: Net,
                     x_train: torch.Tensor,
                     y_train: torch.Tensor,
                     leapfrog_steps: int,
                     steps: int):
    energy_encoder.freeze()

    def energy_function(model_):
        logits = model_(x_train)
        loglikelihood = Categorical(logits=logits).log_prob(y_train).sum()
        return energy_encoder(model_) - loglikelihood

    for _ in range(steps):
        sample(model, target, optimizer, energy_function, leapfrog_steps)

    energy_encoder.unfreeze()
