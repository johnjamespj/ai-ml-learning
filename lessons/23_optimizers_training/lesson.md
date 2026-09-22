# Lesson 23: SGD, momentum and Adam

Gradient descent has a family tree.

## SGD
Update parameters using gradients estimated from a batch rather than necessarily the entire dataset.

## Momentum
Maintain a velocity that accumulates gradient direction, which can smooth noisy updates and accelerate movement along consistent directions.

Conceptually:
v = beta*v + gradient
theta = theta - learning_rate*v

## Adam
Adam tracks moving averages of both gradients and squared gradients and includes bias correction. It adapts effective step sizes per parameter.

## Build it
Extend your NumPy neural network with:
1. mini-batches;
2. SGD;
3. momentum;
4. Adam.

Plot loss curves for each using the same initialization and data.

## Training concepts
Epoch, batch, iteration, learning rate, optimizer state, shuffling, convergence and learning-rate schedules.

## Warning
An optimizer cannot rescue a broken target, leaked dataset or invalid evaluation.
