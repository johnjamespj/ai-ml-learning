# Lesson 06: Probability and statistics for ML

## Concepts
Learn experiments, outcomes, probability, conditional probability, independence, random variables, distributions, expectation, variance, covariance and Bayes' rule.

## Simulation
```python
import numpy as np

rng = np.random.default_rng(42)
samples = rng.normal(loc=10, scale=2, size=10000)

print(samples.mean())
print(samples.var())
print(samples.std())
```

## Conditional probability
P(A|B) means the probability of A given that B occurred.

Bayes' rule:

P(A|B) = P(B|A)P(A) / P(B)

This matters because classifiers often reason in the direction "probability of class given evidence."

## Sampling uncertainty
```python
for n in [10, 100, 1000, 10000]:
    x = rng.normal(0, 1, n)
    print(n, x.mean())
```

## Exercises
1. Simulate 10,000 coin flips.
2. Estimate P(heads).
3. Simulate two dice and estimate P(sum=7).
4. Compare sample means for n=10 and n=10000.
5. Generate two correlated variables and compute covariance/correlation.
6. Explain why correlation does not establish causation.

## ML connection
Probability appears in uncertainty, likelihoods, classifiers and generative models. Statistics tells us what conclusions the finite data actually supports.
