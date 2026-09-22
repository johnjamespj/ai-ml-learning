# Lesson 12: Boosting

Bagging builds many learners largely independently. Boosting builds learners sequentially so later learners focus on errors left by the current ensemble.

## Gradient boosting
Think of the model as an additive sequence:

F_m(x) = F_(m-1)(x) + learning_rate * new_tree(x)

The new learner attempts to improve the current objective.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=200,
    max_leaf_nodes=15,
    random_state=42,
)
model.fit(X_train, y_train)
```

## Concepts
Weak learners, additive models, learning rate, number of estimators, residual/error correction, overfitting and early stopping.

## Exercise
Compare logistic regression, one tree, random forest and gradient boosting on the same split and metrics. Do not select a winner using accuracy alone. Explain the tradeoffs.
