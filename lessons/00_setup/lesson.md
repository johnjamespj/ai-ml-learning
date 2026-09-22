# Lesson 00: Build the laboratory

## Goal
Create a reproducible Python environment and understand the basic workflow used throughout the course.

## Concepts
A Python interpreter executes code. A virtual environment isolates project dependencies. A package is reusable Python software. Jupyter notebooks are excellent for experiments, while normal Python modules are preferable for reusable code.

## Setup
Clone the repository, enter it, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

## Verify
```python
import numpy as np
import pandas as pd
import sklearn
import torch

print(np.__version__)
print(pd.__version__)
print(sklearn.__version__)
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Exercise
Explain in your own words:
1. Why use a virtual environment?
2. What is the difference between a Python package and module?
3. Why might an ML experiment work on one computer and fail on another?
4. Why should dependency versions and random seeds be recorded?

## Done when
You can launch Jupyter, import NumPy/scikit-learn/PyTorch, and explain why reproducibility matters.
