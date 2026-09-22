# Lesson 10: k-nearest neighbors and distance

k-NN predicts using nearby training examples. Training is almost trivial; prediction can be expensive.

## From scratch
```python
import numpy as np

def predict_one(X_train, y_train, x, k=3):
    distances = np.linalg.norm(X_train - x, axis=1)
    nearest = np.argsort(distances)[:k]
    labels, counts = np.unique(y_train[nearest], return_counts=True)
    return labels[np.argmax(counts)]
```

## Key lesson: scale matters
A feature measured in millions can dominate one measured between 0 and 1. Distance-based models often require scaling.

## Experiments
Compare k=1,3,5,15,50. Plot decision boundaries on a 2-D synthetic dataset. Repeat with and without StandardScaler.

## Questions
What happens when k is too small? Too large? What happens in high-dimensional spaces? Why is inference slower than a fitted linear model?
