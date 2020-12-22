import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 bias: bool = True,
                 dropout: float = 0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, input_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        self.out_proj.reset_parameters()

    def forward(self, inputs: torch.Tensor):
        target_length, batch_size, input_dim = inputs.size()
        assert input_dim == self.input_dim
        query, key, value = F.linear(
            inputs, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        query = query * self.scaling
        query = query.contiguous().view(
            target_length, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(
            -1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(
            -1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        attention_output_weights = torch.bmm(query, key.transpose(1, 2))
        expected_size = (batch_size * self.num_heads, target_length, key.size(1))
        assert attention_output_weights.size() == expected_size
        attention_output_weights = F.softmax(attention_output_weights, dim=-1)
        attention_output_weights = F.dropout(attention_output_weights,
                                             p=self.dropout, training=self.training)
        output = torch.bmm(attention_output_weights, value)
        expected_size = (batch_size * self.num_heads, target_length, self.head_dim)
        assert output.size() == expected_size
        output = output.transpose(0, 1).contiguous().view(
            target_length, batch_size, self.embed_dim)
        return F.linear(output, self.out_proj.weight, self.out_proj.bias)


class MultiheadAttentionPool(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.mab = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        nn.init.normal_(self.seed)

    def reset_parameters(self):
        nn.init.normal_(self.seed)
        self.mab._reset_parameters()

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size(1)
        seed = self.seed.repeat(1, batch_size, 1)
        output, _ = self.mab(seed, inputs, inputs, need_weights=False)
        return output.squeeze(0)
