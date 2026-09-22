# Lesson 41: Language modeling

A causal language model learns to predict the next token from earlier tokens.

Given:

"The radar signal is"

training targets might teach the model to predict the next token after each prefix.

## Shifted labels
For token IDs:

[10, 20, 30, 40]

inputs:
[10, 20, 30]

targets:
[20, 30, 40]

## Causal mask
A token must not see future tokens while learning next-token prediction.

## Cross entropy
The output layer produces a distribution over the vocabulary for each position.

## Decoding
Learn:
- greedy decoding
- temperature
- top-k
- top-p/nucleus sampling
- repetition behavior

Decoding changes generated output. It does not retrain the model.

## Exercise
Build a tiny character-level language model on a small text corpus and compare greedy vs sampled generation.
