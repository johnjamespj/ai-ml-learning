# Lesson 60: Monitoring and drift

A deployed model can degrade even if the code never changes.

## Monitor
- input distributions
- missing/invalid inputs
- prediction distributions
- latency/error rate
- confidence/calibration
- labeled performance when labels arrive
- drift

## Drift
Data drift means the input distribution changes. Concept drift means the relationship between inputs and target changes.

## Exercise
Simulate a deployed classifier whose input distribution gradually shifts. Detect the shift and show how model performance changes.

## Lesson
Deployment is an ongoing measurement problem, not a one-time export.
