# Lesson 34: Spectrograms as ML inputs

This lesson connects DSP and deep learning.

## STFT
A short-time Fourier transform analyzes how spectral content changes over time.

```python
from scipy import signal
import numpy as np

fs = 2000
t = np.arange(0, 2, 1/fs)
x = np.sin(2*np.pi*(100 + 200*t)*t)

f, tt, Zxx = signal.stft(x, fs=fs, nperseg=256, noverlap=192)
power = np.abs(Zxx)**2
log_power = 10*np.log10(power + 1e-12)
```

## Questions before ML
- sample rate?
- window?
- segment length?
- overlap?
- frequency resolution?
- time resolution?
- linear or log magnitude?
- normalization method?

These DSP choices become part of the ML data pipeline.

## CNN input
Convert each signal window into a fixed-size time-frequency tensor and feed it to a CNN.

## Synthetic experiment
Generate classes such as:
- tone
- chirp
- broadband pulse
- narrowband interferer

Randomize:
- amplitude
- SNR
- frequency
- duration
- phase
- noise realization

Then train a classifier and test on parameter ranges not used during training.

## Important
Random train/test splitting from the same simulated parameter combinations can give an unrealistically easy result. Hold out meaningful conditions.
