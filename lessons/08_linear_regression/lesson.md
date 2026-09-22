# Lesson 08: Linear regression deeply

## Model
For features X and weights w:

y_hat = Xw + b

The model learns parameters that minimize a loss, commonly mean squared error (MSE).

## From scratch
```python
import numpy as np

rng = np.random.default_rng(42)
X = rng.uniform(-5, 5, (300, 1))
y = 3*X[:, 0] - 4 + rng.normal(0, 2, 300)

w = np.zeros(X.shape[1])
b = 0.0
lr = 0.01

for step in range(2000):
    pred = X @ w + b
    err = pred - y
    loss = np.mean(err**2)

    dw = (2/len(X)) * X.T @ err
    db = 2 * np.mean(err)

    w -= lr * dw
    b -= lr * db

print(w, b, loss)
```

## Residuals
A residual is y - y_hat. Residual plots can reveal curvature, changing variance, outliers, and other structure your model failed to capture.

## Metrics
MSE penalizes large errors strongly. RMSE returns to the target's units. MAE is less sensitive to large errors. R² describes improvement relative to predicting the target mean.

## Experiment
Compare your implementation with `sklearn.linear_model.LinearRegression`. Then add irrelevant features and extreme outliers.

## Questions
Why is linear regression called linear? What assumptions make its interpretation useful? Why can low training error coexist with poor test performance?
