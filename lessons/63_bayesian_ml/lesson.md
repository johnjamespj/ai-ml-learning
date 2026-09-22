# Lesson 63: Bayesian machine learning

## Two views of uncertainty
A frequentist workflow treats model parameters as fixed unknown quantities.

A Bayesian workflow represents uncertainty about parameters using probability distributions.

## Bayes rule
posterior ∝ likelihood × prior

```text
p(theta | data) = p(data | theta) p(theta) / p(data)
```

## Ingredients
- prior: beliefs before observing current data
- likelihood: how compatible data is with parameters
- posterior: updated uncertainty after data
- posterior predictive: uncertainty over future observations

## Coin example
For an unknown coin-bias p, a Beta prior combines neatly with Bernoulli observations.

## Exercise
Implement a Beta-Bernoulli update with SciPy. Plot the prior and posterior after 0, 5, 20 and 200 observations.

## Why this matters
Bayesian reasoning gives a principled language for uncertainty and can be particularly valuable when data is limited or decisions depend on confidence.
