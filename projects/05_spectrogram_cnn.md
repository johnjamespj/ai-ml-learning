# Project 05: Spectrogram CNN

## Mission
Build a complete synthetic signal classification system.

## Signal classes
At minimum:
- single tone
- chirp
- pulsed tone
- narrowband interference

## Randomization
Randomize frequency, phase, amplitude, duration, SNR and noise.

## Pipeline
1. synthesize waveforms;
2. create STFT/spectrogram features;
3. define train/validation/test condition ranges;
4. train a CNN;
5. report class-wise metrics;
6. inspect errors;
7. sweep SNR;
8. plot performance vs SNR.

## Robustness test
Train on one SNR/frequency region and deliberately test on unseen combinations.

## Deliverables
- generator code
- preprocessing code
- PyTorch model
- training script
- evaluation script
- plots
- README with assumptions and findings

## Stretch
Compare the CNN against a classical model trained on hand-engineered spectral features.
