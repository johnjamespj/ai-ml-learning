# Lesson 27: Dataset and DataLoader

Real ML systems rarely fit all training data into one tensor operation.

## Dataset
```python
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## DataLoader
```python
from torch.utils.data import DataLoader

loader = DataLoader(
    SimpleDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
)
```

## Training by mini-batch
```python
for xb, yb in loader:
    logits = model(xb)
    loss = loss_fn(logits, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Concepts
Batch size affects memory use, gradient noise and training throughput.

Shuffling breaks accidental ordering patterns.

Custom Datasets can load files lazily rather than loading everything into memory.

## Exercise
Create a Dataset that stores synthetic signal windows and labels. Return:
- waveform tensor
- signal class
- SNR metadata

Then build a DataLoader and inspect one batch.
