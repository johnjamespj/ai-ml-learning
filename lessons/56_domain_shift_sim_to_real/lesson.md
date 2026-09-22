# Lesson 56: Domain shift and sim-to-real

Models often fail because deployment data differs from training data.

## Types of shift
- covariate shift
- label shift
- concept drift
- sensor changes
- environment changes
- simulation mismatch

## Signal examples
A synthetic RF model may encounter:
- different noise statistics
- receiver nonlinearities
- clipping
- frequency response
- timing offsets
- phase noise
- interference not present in simulation

## Domain randomization
Randomize plausible nuisance variables during training.

## Exercise
Train a classifier on ideal synthetic signals. Then progressively add receiver distortion, colored noise, frequency offset and clipping at test time.

Plot the performance degradation.

## Lesson
Robustness must be designed and measured. It does not appear automatically because the validation split was high-scoring.
