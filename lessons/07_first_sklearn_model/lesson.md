# Lesson 07: Your first complete scikit-learn experiment

## Goal
Train, evaluate and interrogate a model without contaminating the test set.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

## The experiment
The test set represents unseen data. Do not repeatedly tune decisions against it. The pipeline also ensures scaling is learned from training data rather than leaking test-set information.

## Vocabulary
- **fit**: estimate model parameters from data
- **predict**: produce outputs for inputs
- **hyperparameter**: configuration chosen outside training
- **generalization**: performance on unseen data
- **baseline**: simple reference that a more complicated system should beat

## Exercises
1. Record train and test accuracy.
2. Remove scaling and compare.
3. Change the split seed and observe variation.
4. Explain every cell in the confusion matrix.
5. Explain why accuracy can be dangerous for heavily imbalanced classes.
6. Identify parameters vs hyperparameters in this experiment.
