# Lesson 66: Ensembles, voting and stacking

Different models make different errors. Ensembles attempt to combine them.

## Methods
- bagging
- boosting
- hard voting
- soft voting
- stacking

## Stacking
Base models generate predictions that become inputs to a meta-model.

Critical rule: training predictions for the meta-model must be generated out-of-fold, otherwise leakage is severe.

## Exercise
Combine logistic regression, random forest and gradient boosting:
1. compare individual models;
2. soft-vote probabilities;
3. build a leakage-safe stacked model;
4. compare error overlap.

## Lesson
An ensemble helps most when its members have useful but imperfectly correlated errors.
