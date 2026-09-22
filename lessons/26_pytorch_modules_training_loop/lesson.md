# Lesson 26: nn.Module and the training loop

## Goal
Move from raw tensors to reusable neural-network components.

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)
```

## Canonical training loop
```python
model = MLP(2, 32, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()

    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## What each line means
- `model.train()`: training behavior
- forward pass: compute predictions
- loss: measure objective
- zero gradients: clear previous accumulation
- backward: compute gradients
- step: update parameters

## Why gradients accumulate
PyTorch adds new gradients to existing `.grad` values. This is useful for some workflows but dangerous if you forget to clear them.

## Exercise
Rewrite the loop with SGD. Compare SGD, momentum and Adam using identical initial weights and data.
