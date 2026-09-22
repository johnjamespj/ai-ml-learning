# Lab 01: Linear Regression From Scratch

Linked lessons: 01, 05, 08.

## Build
Implement:
- prediction
- mean squared error
- analytical gradients
- gradient-descent training

## Run
```bash
python labs/lab01_linear_regression/starter.py
LAB_TARGET=starter pytest labs/lab01_linear_regression/test_lab.py -q
```

## Experiments
1. Change the learning rate across 0.0001, 0.001, 0.01, 0.1.
2. Add Gaussian noise.
3. Add extreme outliers.
4. Compare learned coefficients with scikit-learn.

## Explain
Without code, explain why the sign of the gradient tells you which direction to move a parameter and why a learning rate can make training diverge.
