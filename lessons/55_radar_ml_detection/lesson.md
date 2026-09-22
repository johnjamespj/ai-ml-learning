# Lesson 55: Radar and RF detection with ML

## Detection framing
Before training, define the task precisely.

Examples:
- signal present vs absent
- interference type
- emitter class
- anomaly vs expected background
- target class
- propagation condition

## Operating metrics
For detection tasks, examine:
- probability of detection
- false alarm probability/rate
- precision/recall
- ROC
- precision-recall curve
- calibration
- performance vs SNR

## Threshold sweep
A classifier score is not the final decision. Sweep thresholds and study operating points.

## Experiment
Build a synthetic detector where SNR spans -20 dB to +20 dB.

Report:
- Pd vs SNR
- false-alarm behavior
- confusion matrix
- calibration
- generalization to held-out frequencies

## Critical lesson
A single aggregate accuracy number hides the operating behavior engineers actually care about.
