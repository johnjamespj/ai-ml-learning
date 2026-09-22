# Lesson 04: Linear algebra for ML

## Why it matters
A large fraction of ML can be described as vectors flowing through matrix operations.

## Vectors and dot products
```python
import numpy as np

x = np.array([2., 3., 4.])
w = np.array([0.5, -1., 2.])

manual = sum(x[i] * w[i] for i in range(len(x)))
vectorized = x @ w
print(manual, vectorized)
```

The dot product is a weighted combination and also relates to vector similarity.

## Matrix multiplication
For n samples and d features:
- X has shape (n,d)
- w has shape (d,)
- X @ w has shape (n,)

```python
X = np.array([[1.,2.], [3.,4.], [5.,6.]])
w = np.array([0.2, 0.8])
print(X @ w)
```

## Norms
```python
v = np.array([3.,4.])
print(np.linalg.norm(v))
```

Norms measure magnitude and later appear in distance metrics and regularization.

## Eigenvectors intuition
For some special directions v, a matrix transformation changes magnitude but not direction:

A v = lambda v

```python
A = np.array([[2., 0.], [0., 1.]])
values, vectors = np.linalg.eig(A)
print(values)
print(vectors)
```

This becomes important in PCA and many other methods.

## Exercises
1. Compute a dot product manually and with NumPy.
2. Predict the shapes produced by five matrix multiplications before running them.
3. Normalize a vector to unit length.
4. Calculate Euclidean distance between two vectors.
5. Explain what X, w and X@w mean in a linear model.
