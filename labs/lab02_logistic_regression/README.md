# Lab 02: Logistic Regression From Scratch

Linked lessons: 09 and 15.

Implement sigmoid, binary cross-entropy, gradients, fit, probability prediction, and thresholded classification.

## Tests
```bash
LAB_TARGET=starter pytest labs/lab02_logistic_regression/test_lab.py -q
```

## Experiments
Create an imbalanced dataset and sweep thresholds from 0.05 to 0.95. Plot precision and recall versus threshold.

## Explain
Why is 0.5 only a convention? Give a detection example where a lower threshold would be reasonable and describe its cost.
