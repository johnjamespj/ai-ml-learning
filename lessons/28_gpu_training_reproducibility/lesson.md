# Lesson 28: GPU training and reproducibility

## Device-aware code
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)

    logits = model(xb)
```

## GPU mental model
A GPU is valuable when the workload contains large amounts of parallel numerical work. Moving tiny arrays back and forth can cost more than the computation itself.

## Reproducibility
```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

Exact determinism can depend on hardware, kernels and library settings.

## Measure, do not assume
Time:
- data loading;
- host-to-device transfer;
- forward pass;
- backward pass;
- optimizer step.

## Exercise
Benchmark the same MLP on CPU and GPU at increasing batch sizes. Explain where GPU acceleration starts becoming useful.
