# Lesson 13: Support vector machines and kernels

A linear SVM searches for a separating hyperplane with a large margin.

## Intuition
The closest influential training examples are support vectors. The regularization parameter C controls the tradeoff between a wide margin and classification violations.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(
    StandardScaler(),
    SVC(C=1.0, kernel="rbf", gamma="scale", probability=True)
)
model.fit(X_train, y_train)
```

## Kernel idea
A kernel lets the algorithm behave as if data were represented in a richer feature space without explicitly constructing every transformed feature.

## Experiments
Use a nonlinear 2-D dataset. Compare linear and RBF kernels. Sweep C and gamma and visualize decision boundaries.

## Questions
Why does scaling matter? What does C change? What does gamma change for an RBF kernel? Why can a highly flexible boundary overfit?
