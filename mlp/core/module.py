from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.attention import MultiheadAttentionPool, SelfAttention


class Net(nn.Module, ABC):
    @abstractmethod
    def reset_parameters(self) -> None:
        pass

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class LeNet(Net):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, inputs: torch.Tensor):
        output = self.pool(F.relu(self.conv1(inputs)))
        output = self.pool(F.relu(self.conv2(output)))
        output = output.view(-1, 16 * 5 * 5)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


class EnergyEncoder(Net):
    def __init__(self,
                 embed_dim: int,
                 input_shape: torch.Size,
                 num_heads: int,
                 num_states: int,
                 out_dim: int,
                 ):
        super().__init__()
        self.probes = nn.Parameter(torch.Tensor(num_states, *input_shape))
        self.mha1 = SelfAttention(out_dim, embed_dim, num_heads)
        self.mha2 = SelfAttention(embed_dim, embed_dim, num_heads)
        self.pma = MultiheadAttentionPool(embed_dim, num_heads)
        self.dense = nn.Linear(embed_dim, 1)
        self.out_dim = out_dim
        nn.init.normal_(self.probes)

    def reset_parameters(self):
        nn.init.normal_(self.probes)
        self.mha1.reset_parameters()
        self.mha2.reset_parameters()
        self.pma.reset_parameters()
        self.dense.reset_parameters()

    def forward(self, state: nn.Module):
        output = F.relu(state(self.probes))
        output = output.view(-1, 1, self.out_dim)
        output = self.mha1(output)
        output = self.mha2(output)
        output = self.pma(output)
        return self.dense(output)


class EnergyEncoderDeepSet(Net):
    def __init__(self,
                 embed_dim: int,
                 input_shape: torch.Size,
                 num_states: int,
                 out_dim: int,
                 ):
        super().__init__()
        self.probes = nn.Parameter(torch.Tensor(num_states, *input_shape))
        self.dense1 = nn.Linear(out_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, 1)
        self.out_dim = out_dim
        nn.init.normal_(self.probes)

    def reset_parameters(self):
        nn.init.normal_(self.probes)
        self.dense1.reset_parameters()
        self.dense2.reset_parameters()

    def forward(self, state: nn.Module):
        output = F.relu(state(self.probes))
        output = output.view(-1, 1, self.out_dim)
        output = F.relu(self.dense1(output))
        output = output.mean(dim=0, keepdim=True)
        return self.dense2(output)
