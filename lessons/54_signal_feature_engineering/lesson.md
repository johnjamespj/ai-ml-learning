# Lesson 54: Signal feature engineering for ML

Before deep learning, build strong signal-domain features.

## Time-domain features
Examples:
- RMS
- peak
- crest factor
- zero-crossing rate
- skewness
- kurtosis

## Frequency-domain features
Examples:
- dominant frequency
- spectral centroid
- bandwidth
- bandpower
- spectral entropy
- peak count

## Time-frequency features
Examples:
- STFT statistics
- spectrogram patch statistics
- wavelet coefficients

## Example bandpower
```python
import numpy as np
from scipy.signal import welch

f, Pxx = welch(x, fs=fs, nperseg=1024)
mask = (f >= f1) & (f <= f2)
bandpower = np.trapz(Pxx[mask], f[mask])
```

## Exercise
Generate tone/chirp/noise classes and build a classical classifier from engineered features before training a neural network.

## Lesson
Hand-engineered features give a strong interpretable baseline and can remain competitive when data is limited.
