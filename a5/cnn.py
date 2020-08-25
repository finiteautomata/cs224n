#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i

class CNN(nn.Module):

    def __init__(self, input_dim, num_filters, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, num_filters, kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input is shape (batch_size, emb size, word length)
        conv_out = F.relu(self.conv(input))
        # conv_out is (batch_size, num_filters, word_length-num_filters+1)
        max_out, _ = torch.max(conv_out, 2)

        return max_out

### END YOUR CODE
