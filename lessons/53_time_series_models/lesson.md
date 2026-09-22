# Lesson 53: ML and deep learning for time series

## Classical approaches
Start with autoregressive and feature-engineered models.

## Deep approaches
Compare:
- MLP on fixed windows
- 1-D CNN
- LSTM/GRU
- Transformer-style temporal model

## Important
A more expressive sequence model is not automatically better than a strong baseline.

## Evaluation
Use rolling or walk-forward validation.

Measure metrics at the required forecast horizon rather than mixing all horizons together.

## Exercise
Build the same forecasting task using:
1. persistence baseline;
2. linear regression on lags;
3. 1-D CNN;
4. LSTM.

Compare complexity, training time and validation error.
