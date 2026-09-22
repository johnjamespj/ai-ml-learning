# Lesson 19: Neural networks from first principles

## A neuron
A neuron computes:

z = x @ w + b

a = activation(z)

A layer applies many neurons in parallel:

Z = XW + b

## Why nonlinear activation matters
Without nonlinear activations, stacking linear layers still collapses into one linear transformation.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

X = np.array([[1.,2.],[3.,4.]])
W = np.array([[0.2,-0.3,0.8],[0.5,0.1,-0.2]])
b = np.zeros(3)

Z = X @ W + b
A = relu(Z)
print(Z)
print(A)
```

## Architecture vocabulary
Input layer, hidden layer, output layer, weights, biases, activation, width and depth.

## Exercise
Build a two-layer forward pass with NumPy:
X -> Linear -> ReLU -> Linear -> output

Predict every intermediate array shape before running the code.

## Core mental model
A neural network is a parameterized function. Training searches parameter space for values that make its outputs useful according to an objective.
