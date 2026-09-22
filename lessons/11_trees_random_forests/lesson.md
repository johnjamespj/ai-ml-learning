# Lesson 11: Decision trees and random forests

A decision tree recursively splits feature space to make child groups purer.

## Classification tree
```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
```

Learn entropy, Gini impurity, information gain, depth, leaves, and pruning.

## Why trees overfit
A sufficiently deep tree can memorize tiny regions of training space.

## Random forests
A random forest trains many randomized trees and aggregates them. Bagging reduces instability/variance.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
forest.fit(X_train, y_train)
```

## Experiments
Sweep tree depth. Compare train/test performance. Then compare one tree with a random forest.

## Caution
Feature importance is useful but can be misleading. Importance is not causality.
