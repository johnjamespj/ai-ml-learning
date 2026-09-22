# Lesson 25: PyTorch tensors and autograd

## Goal
Understand what PyTorch adds on top of NumPy.

PyTorch tensors look similar to NumPy arrays, but they can:
- live on CPUs or GPUs;
- track operations;
- automatically compute gradients.

## Tensors
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
W = torch.randn(3, 2)
b = torch.zeros(2)

y = x @ W + b
print(y)
print(y.shape)
```

## Devices
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
W = W.to(device)
```

A tensor and the tensors it interacts with generally need to be on the same device.

## Autograd
```python
w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(2.0)
y = (w*x - 5)**2

y.backward()
print(w.grad)
```

PyTorch builds a computational graph during the forward pass and uses reverse-mode automatic differentiation during `.backward()`.

## Compare against manual calculus
For:

L=(wx-5)^2

derive dL/dw by hand, evaluate it at w=3 and x=2, then compare against PyTorch.

## Critical methods
- `requires_grad=True`
- `.backward()`
- `.grad`
- `.detach()`
- `torch.no_grad()`
- `.zero_grad()`

## Exercise
Rebuild your Lesson 01 linear regression using torch tensors and manual parameter updates while letting autograd calculate gradients.
