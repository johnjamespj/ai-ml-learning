# Lesson 73: Data-centric AI

Model architecture is only one lever. Often the largest gains come from improving the data.

## Investigate
- incorrect labels
- duplicates
- class imbalance
- ambiguous examples
- bad sensors
- inconsistent preprocessing
- missing coverage
- train/deployment mismatch

## Error slicing
Do not inspect only aggregate metrics. Break performance down by meaningful slices such as:
- SNR
- frequency
- device
- geography
- class
- environment

## Exercise
Inject label noise and duplicate leakage into a dataset. Observe metric changes, clean the data and compare the benefit with switching to a more complex model.

## Lesson
Better data can beat a fancier algorithm.
