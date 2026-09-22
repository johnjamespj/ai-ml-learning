# Lesson 64: Uncertainty and calibration

A classifier that outputs 0.9 is useful only if its confidence has meaning.

## Calibration
Among predictions assigned about 90% confidence, roughly 90% should be correct in a well-calibrated system.

## Measure
Study:
- reliability diagrams
- Brier score
- log loss
- expected calibration error concepts

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_true, y_prob, n_bins=10
)
```

## Sources of uncertainty
- aleatoric: irreducible noise in observations
- epistemic: uncertainty due to limited knowledge/model/data

## Exercise
Train a classifier, intentionally overfit it, then inspect accuracy and calibration separately. Try probability calibration on a held-out calibration set.

## Detection connection
Threshold selection depends heavily on whether scores remain meaningful under SNR and domain shift.
