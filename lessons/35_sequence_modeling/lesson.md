# Lesson 35: Sequence modeling

Many problems have ordered inputs:
- text tokens
- audio samples
- sensor streams
- radar pulses
- time-series measurements

Order matters.

## Sliding windows
A simple sequence model can begin by converting a long series into fixed windows.

```python
import numpy as np

def make_windows(x, window):
    X, y = [], []
    for i in range(len(x)-window):
        X.append(x[i:i+window])
        y.append(x[i+window])
    return np.array(X), np.array(y)
```

## Questions
What information is lost if sequence order is shuffled? What is the difference between sequence classification, sequence-to-sequence modeling and next-step prediction?

## Baseline first
Before using an RNN, compare against:
- last-value prediction
- moving average
- linear/autoregressive baseline
- MLP on flattened windows

A sequence model should beat a meaningful simpler baseline.
