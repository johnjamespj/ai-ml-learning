import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(16*7*7,64),nn.ReLU(),nn.Linear(64,num_classes))

    def forward(self,x):
        return self.classifier(self.features(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
