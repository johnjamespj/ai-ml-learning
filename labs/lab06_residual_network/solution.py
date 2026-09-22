import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d))

    def forward(self,x):
        return torch.relu(x+self.net(x))
