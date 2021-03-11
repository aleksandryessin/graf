
import torch
from torch import nn
import torch.nn.functional as F

import math

class SineLayer(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            w0 = 1.0, 
            c = 6.0, 
            is_first = False, 
            bias = True, 
            activation_on = True,
            initializer = "uniform"
        ):

        super().__init__()
        self.activation_on = activation_on
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.initializer = initializer
        self.c = c
        self.w0 = w0

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weigths()

def init_weigths(self):
    dim, c, w0 = self.in_features, self.c, self.w0

    with torch.no_grad():
        if self.is_first:
            self.linear.weight.uniform_(-1 / dim, 1 / dim)
        else:
            self.linear.weight.uniform_(-math.sqrt(c / dim) / w0, math.sqrt(c / dim) / w0)

def forward(self, x, gamma=None, beta=None):
    out = self.linear(x)

    # FiLM modulation
    if gamma is not None:
        out = out * gamma

    if beta is not None:
        out = out + beta

    if self.activation_on:
        return torch.sin(self.w0 * out)
    else:
        return self.w0 * out