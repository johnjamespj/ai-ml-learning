# Lab 04: NumPy MLP and Backpropagation

Linked lessons: 19-24.

## Mission
Implement a 2-layer classifier:
X -> Linear -> ReLU -> Linear -> Softmax

You must implement the forward pass, cross-entropy, backward pass and parameter update.

## Debugging gate
Before training a full dataset, make the network overfit four XOR-like samples.

## Tests
```bash
LAB_TARGET=starter pytest labs/lab04_numpy_mlp/test_lab.py -q
```

## Explain
For every gradient matrix, state its shape and why that shape is required.
