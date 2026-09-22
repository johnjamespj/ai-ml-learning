# Project 10: Radar / Signal ML System

## Mission
Build an end-to-end RF or radar-style ML experiment.

## Data
Synthetic first. Generate configurable:
- tones
- chirps
- pulses
- interference
- noise
- frequency offsets
- amplitude/SNR variation

## Baselines
1. threshold/statistical detector
2. engineered-feature classical ML model
3. neural model on waveform or spectrogram

## Evaluation
Report:
- Pd vs SNR
- false-alarm behavior
- ROC/PR curves
- held-out frequency ranges
- robustness to noise model shift
- robustness to clipping or receiver distortion

## Engineering report
Explain:
- sample rate
- bandwidth
- window/STFT configuration
- normalization
- split strategy
- model
- latency
- failure cases
- deployment assumptions
