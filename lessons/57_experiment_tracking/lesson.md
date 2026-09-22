# Lesson 57: Experiment tracking and reproducibility

A useful experiment should be reconstructible after you forget the details.

## Track
- code commit
- dataset version
- split/version
- random seed
- model architecture
- hyperparameters
- preprocessing
- metrics
- artifacts
- hardware/software environment

## Minimal approach
Store configuration as YAML/JSON and write metrics to structured files.

## Tools
Learn the role of experiment trackers such as MLflow or Weights & Biases, but first understand the information being tracked.

## Exercise
Run a hyperparameter sweep and produce a table linking each run to:
- config
- seed
- metric
- saved model
- commit hash

## Rule
A result without provenance is difficult to trust.
