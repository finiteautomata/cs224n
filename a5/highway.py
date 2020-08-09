#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
from torch import nn
from torch.nn import functional as F
### YOUR CODE HERE for part 1h

class Highway(nn.Module):

    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input (batch_size, dim)
        x_proj = F.relu(self.proj(input))

        x_gate = torch.sigmoid(self.gate(input))
        # x_proj and x_gate (batch_size, dim)

        x_highway = x_gate * x_proj + (1 - x_gate) * input

        return self.dropout(x_highway)

### END YOUR CODE
