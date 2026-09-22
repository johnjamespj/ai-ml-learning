# Lesson 17: PCA and dimensionality reduction

## Goal
Understand how high-dimensional data can be represented using fewer directions.

PCA finds orthogonal directions of maximum variance. The first principal component captures the greatest possible variance, the second captures the greatest remaining variance subject to orthogonality, and so on.

## From covariance to components
```python
import numpy as np

X = np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0]])
Xc = X - X.mean(axis=0)
cov = np.cov(Xc, rowvar=False)
values, vectors = np.linalg.eigh(cov)
order = np.argsort(values)[::-1]
values = values[order]
vectors = vectors[:, order]
Z = Xc @ vectors
print(values)
print(Z)
```

## With scikit-learn
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
```

## Important
PCA preserves variance, not labels, causality, or necessarily the information most useful for prediction.

## Exercises
Standardize a dataset, fit PCA, plot cumulative explained variance, reconstruct samples using fewer components and measure reconstruction error.

## Signal connection
Spectra and time-frequency features can be extremely high-dimensional. PCA is one tool for exploring correlated structure before moving to learned representations.
