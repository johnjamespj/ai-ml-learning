# Lesson 32: Build and train a CNN

## Model
```python
import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

## Assignment
Train on MNIST or FashionMNIST.

Track:
- train loss
- validation loss
- accuracy
- confusion matrix
- per-class accuracy

## Experiments
Change:
- kernel size
- number of channels
- depth
- pooling
- learning rate

Predict which change alters parameter count before running it.

## Question
Why can a CNN generalize better on images than a huge fully connected network with similar capacity?
