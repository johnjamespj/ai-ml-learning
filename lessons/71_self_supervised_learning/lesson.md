# Lesson 71: Self-supervised learning

Labeling is expensive. Self-supervised learning creates learning signals from the data itself.

## Examples
- predict masked tokens
- predict next token
- contrast augmented views
- reconstruct corrupted inputs

## Contrastive intuition
Represent two related views near each other in embedding space while separating unrelated samples.

## Why it matters
A representation can be pretrained on large unlabeled corpora and later adapted to smaller labeled tasks.

## Signal connection
Possible pretext tasks:
- reconstruct masked waveform segments
- identify whether two windows came from the same recording
- contrast augmented spectrogram views

## Exercise
Build a small contrastive or reconstruction pretraining experiment, then compare downstream classification with and without pretraining.
