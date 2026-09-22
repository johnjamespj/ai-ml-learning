# Lesson 03: Visualization and exploratory data analysis

## Goal
Learn to inspect data before choosing a model.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 200)
y = 2*x + 1 + rng.normal(0, 2, len(x))

plt.scatter(x, y, alpha=0.6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic observations")
plt.show()
```

Explore distributions, relationships, outliers, class balance and suspicious artifacts.

## Histogram
```python
plt.hist(y, bins=20)
plt.show()
```

## Exercise
Generate two classes with different means and overlapping noise. Plot:
1. each feature distribution;
2. feature-vs-feature scatter;
3. class counts;
4. a correlation matrix using NumPy/pandas and Matplotlib.

Write five observations before fitting any model.

## Why this matters
Models happily learn bad data. EDA is where you notice that one sensor is saturated, one class has only 12 examples, or a feature accidentally contains the answer.
