# Lab 10: Signal Detection Baseline

Linked lessons: 54-56.

Before using ML, build a classical detector.

## Task
Generate noisy observations that either contain or do not contain a sinusoid. Build a quadrature matched-filter score that is insensitive to random phase.

## Evaluation
1. Calibrate a threshold using noise-only validation data.
2. Choose a target false-alarm rate.
3. Estimate probability of detection across SNR.
4. Plot Pd vs SNR.

## Tests
```bash
LAB_TARGET=starter pytest labs/lab10_signal_detection/test_lab.py -q
```

## ML extension
Train logistic regression and a small neural model on the same raw/engineered data. They must beat this baseline for a reason, not merely exist.
