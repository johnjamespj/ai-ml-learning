# Lesson 22: Build a neural network with only NumPy

## Mission
Build and train a classifier without PyTorch, TensorFlow or scikit-learn estimators.

Required architecture:
2 inputs -> hidden layer -> ReLU -> output layer -> softmax

## Required components
Implement:
- parameter initialization
- forward pass
- ReLU
- stable softmax
- cross-entropy
- backward pass
- SGD parameter update
- prediction
- accuracy
- training loop
- loss history

Use `sklearn.datasets.make_moons` only to generate data.

## Training loop skeleton
```python
for epoch in range(epochs):
    logits, cache = forward(X, params)
    loss = cross_entropy(logits, y)
    grads = backward(X, y, params, cache)

    for name in params:
        params[name] -= learning_rate * grads[name]
```

Your exact gradient naming may differ.

## Experiments
Change hidden width, learning rate, initialization scale, number of epochs and noise level. Plot decision boundaries and training loss.

## Debugging rules
Check shapes first. Check for NaN/Inf. Verify loss decreases on a tiny dataset. Overfit a very small batch deliberately. Gradient-check parameters.

## Milestone
Once this works, PyTorch's autograd will no longer look magical. It automates machinery you have now built.
