# Lesson 21: Backpropagation

This is a major milestone.

## What backprop actually does
A forward pass computes predictions and loss. Backpropagation efficiently applies the chain rule backward through the computational graph to calculate how each parameter affects that loss.

Consider:

Z1 = XW1 + b1
A1 = ReLU(Z1)
Z2 = A1W2 + b2

For MSE, begin with the derivative of loss with respect to Z2, then propagate gradients backward.

## Matrix gradient structure
```python
# schematic backward pass
dZ2 = 2 * (Z2 - y) / len(X)
dW2 = A1.T @ dZ2
db2 = dZ2.sum(axis=0)

dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)
dW1 = X.T @ dZ1
db1 = dZ1.sum(axis=0)
```

## Gradient checking
Numerically perturb a parameter and compare the numerical derivative with your analytical backprop gradient. This is one of the best ways to debug a from-scratch network.

## Exercise
Draw the computational graph by hand. Annotate every tensor shape on the forward and backward pass. Then implement gradient checking for several random weights.

## Done when
You can explain why each matrix multiplication in the backward pass has its particular shape.
