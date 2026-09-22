# Lesson 09: Logistic regression and classification

Linear regression predicts unrestricted numbers. Binary classification needs a score that can be interpreted as a probability.

## Sigmoid
sigma(z) = 1 / (1 + exp(-z))

```python
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

z = np.linspace(-8, 8, 9)
print(sigmoid(z))
```

Logistic regression computes z=Xw+b and maps it through sigmoid.

## Decision boundary
A probability threshold converts probability to a class. 0.5 is common, not sacred. Changing the threshold changes false positives and false negatives.

## Loss
Binary cross entropy rewards calibrated probability assigned to the correct outcome:

L = -mean(y log(p) + (1-y) log(1-p))

## From scratch challenge
Implement sigmoid, binary cross entropy, gradients, gradient descent, `predict_proba`, and thresholded `predict`.

Then compare with `sklearn.linear_model.LogisticRegression`.

## Important
Classification is not merely "correct vs wrong." In detection problems, the costs of missed detections and false alarms may be radically different.
