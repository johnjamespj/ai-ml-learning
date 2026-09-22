# Project 03: Neural Network From Scratch

## Rule
No deep-learning framework for the model or gradients.

Allowed: NumPy, Matplotlib, and scikit-learn only for dataset generation/splitting/metrics.

## Build
Create a reusable `MLP` implementation with:
- configurable hidden width
- ReLU
- stable softmax
- cross entropy
- forward pass
- backpropagation
- mini-batches
- SGD and one advanced optimizer
- deterministic initialization
- training history
- predict/predict_proba

## Verification
1. Gradient-check selected parameters.
2. Overfit a tiny dataset.
3. Train on make_moons.
4. Plot decision boundary.
5. Plot train/validation loss.
6. Report a confusion matrix.
7. Explain at least three failure modes.

## Stretch
Implement another hidden layer without copying the entire training algorithm. Refactor toward a list of layers.

## Report
Explain the path from X to logits and from loss back to each weight matrix. If you can explain that clearly, you understand the core mechanics underneath modern deep learning.
