import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

def make_data(n=256, seed=0):
    # TODO: generate a simple binary classification problem
    raise NotImplementedError

def train_step(model, X, y, optimizer, loss_fn):
    # TODO
    raise NotImplementedError

def accuracy(model, X, y):
    # TODO
    raise NotImplementedError
