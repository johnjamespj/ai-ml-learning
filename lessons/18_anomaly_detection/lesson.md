# Lesson 18: Anomaly detection

Anomaly detection asks whether an observation is unusual relative to expected behavior.

## Why it is tricky
Anomalies are often rare, labels may be scarce, and "unusual" is not identical to "bad."

## Methods
Start with simple statistical thresholds, then compare Isolation Forest and Local Outlier Factor.

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.02, random_state=42)
prediction = model.fit_predict(X)
# -1 indicates an anomaly, +1 an inlier
```

## Build a baseline
For one-dimensional data, begin with a z-score or robust median/MAD detector. Complex algorithms should earn their complexity.

## Experiment
Generate normal measurements plus injected anomalies. Sweep the anomaly threshold/contamination assumption. Measure precision and recall against known injected anomalies.

## Signal connection
Potential applications include interference detection, sensor faults, unexpected spectra and changing environmental conditions. The hard part is defining normal behavior robustly.

## Questions
Why is accuracy especially poor for anomaly detection? What happens when normal behavior changes over time? What is concept drift?
