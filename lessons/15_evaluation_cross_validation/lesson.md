# Lesson 15: Evaluation, metrics and cross-validation

This lesson is one of the most important in the repository. A powerful model with a broken evaluation is a broken experiment.

## Classification metrics
From TP, TN, FP and FN:

precision = TP/(TP+FP)

recall = TP/(TP+FN)

F1 = harmonic mean of precision and recall

Also learn specificity, ROC/AUC, precision-recall curves, log loss and calibration.

## Regression metrics
MAE, MSE, RMSE and R² answer different questions.

## Cross-validation
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    model, X, y, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

for metric, values in scores.items():
    if metric.startswith("test_"):
        print(metric, values.mean(), values.std())
```

## Data splitting traps
Random splitting is not always valid. Time-series data often requires chronological splitting. Multiple measurements from the same subject/device/scene may require grouped splitting. Otherwise closely related examples can leak across folds.

## Detection connection
For radar/signal detection, examine probability of detection and false alarm behavior, thresholds, class imbalance and operating points rather than treating accuracy as sufficient.

## Exercise
Construct a dataset with 98% negatives. Show how a dummy classifier can achieve impressive accuracy while being useless for detecting positives. Compare accuracy, recall, precision, F1, ROC-AUC and PR-AUC.
