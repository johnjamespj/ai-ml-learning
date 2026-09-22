# Lesson 65: Hyperparameter optimization

Hyperparameters are choices made outside normal parameter fitting.

Examples:
- learning rate
- regularization strength
- tree depth
- number of layers
- batch size

## Search methods
- manual search
- grid search
- random search
- Bayesian optimization concepts

## Why random search can beat grids
If only a few dimensions strongly affect performance, random search explores more distinct values of those dimensions.

## Correct workflow
Never tune on the final test set.

Use:
training data -> cross-validation/tuning -> chosen model -> one final test evaluation

## Exercise
Compare GridSearchCV and RandomizedSearchCV on the same model and computational budget.

## Advanced concept
Optimization itself can overfit the validation procedure when you run enough experiments. Keep a final untouched test set.
