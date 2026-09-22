# Lesson 01: NumPy and the language of ML

Machine learning is mostly operations on collections of numbers. NumPy gives Python efficient multidimensional arrays and vectorized numerical operations.

## 1. Scalars, vectors, matrices and tensors
A scalar is one number. A vector is a 1-D array. A matrix is a 2-D array. In ML, tensor commonly means an n-dimensional array.

```python
import numpy as np

scalar = 3.0
x = np.array([1., 2., 3.])
A = np.array([[1., 2.], [3., 4.]])
T = np.zeros((4, 3, 32, 32))

print(x.shape, A.shape, T.shape)
```

Ask yourself what every axis represents. For ML data, a matrix is often shaped `(samples, features)`.

## 2. Vectorization
Avoid manually looping over every numeric element when an array operation expresses the same calculation.

```python
x = np.array([1., 2., 3., 4.])
print(x * 2)
print(x ** 2)
print(x.mean())
print(x.std())
```

## 3. Dot product
```python
x = np.array([1., 2., 3.])
w = np.array([0.5, -1., 2.])
print(np.dot(x, w))
```

The weighted sum `x @ w` appears everywhere in ML. A linear model is essentially

`y_hat = X @ w + b`

## 4. Tiny model from scratch
```python
X = np.array([[1.], [2.], [3.], [4.]])
y = np.array([3., 5., 7., 9.])

w = 0.0
b = 0.0
lr = 0.01

for step in range(2000):
    pred = X[:, 0] * w + b
    error = pred - y
    loss = np.mean(error ** 2)

    dw = 2 * np.mean(error * X[:, 0])
    db = 2 * np.mean(error)

    w -= lr * dw
    b -= lr * db

print(w, b, loss)
```

You just trained a model. The model has parameters `w,b`; predictions come from a forward calculation; MSE measures error; derivatives tell us how parameters affect the error; gradient descent changes the parameters to reduce it.

## Exercises
1. Predict the learned output for x=10.
2. Change the target to `y = 4x - 3` and retrain.
3. Change the learning rate to 0.0001, 0.1 and 1.0. Record what happens.
4. Rewrite the prediction using matrix multiplication.
5. Plot training loss versus iteration.

## Challenge
Generate noisy observations of `y=2x+1`. Train your implementation and compare its coefficients with `sklearn.linear_model.LinearRegression`.

## Questions you should be able to answer
What is a feature? What is a parameter? What is a prediction? What is a loss function? What is a gradient? What does the learning rate control?
