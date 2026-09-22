# Lesson 20: Activations and loss functions

## Activations
Study sigmoid, tanh, ReLU, leaky ReLU and softmax.

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
```

Subtracting the maximum in softmax improves numerical stability.

## Losses
Regression commonly uses MSE/MAE. Binary classification commonly uses binary cross entropy. Multiclass classification commonly uses cross entropy.

A loss is the training objective, not automatically the metric you ultimately care about.

## Experiment
Plot each activation and its derivative. Examine saturation. Feed increasingly large values to naive vs stable softmax.

## Questions
Why do hidden layers need nonlinearity? Why is softmax useful for mutually exclusive classes? Why can optimizing one loss still require evaluating several real-world metrics?
