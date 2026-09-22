# Lesson 52: Time-series foundations

## Time changes the split
A random split often leaks future information into training.

Prefer chronological splitting when forecasting future behavior.

## Concepts
- trend
- seasonality
- autocorrelation
- stationarity
- lag features
- rolling statistics
- forecasting horizon

## Baselines
Always compare against:
- last value
- moving average
- seasonal naive baseline where appropriate

## Example lag features
```python
import pandas as pd

df["lag_1"] = df["value"].shift(1)
df["lag_5"] = df["value"].shift(5)
df["rolling_mean_10"] = df["value"].rolling(10).mean()
```

## Exercise
Generate a noisy sinusoidal signal with drift. Build lag features and compare a linear model against a naive baseline.
