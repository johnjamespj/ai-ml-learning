# Lesson 67: Interpretability and explainability

## Different questions
- How does the model work globally?
- Why did it make this specific prediction?
- Which input changes affect output?
- Is the explanation stable?

## Tools/concepts
- coefficients
- tree inspection
- permutation importance
- partial dependence
- SHAP concepts
- saliency for neural networks

## Caution
An explanation method explains model behavior, not necessarily the real-world causal mechanism.

## Exercise
For one tabular classifier:
1. inspect coefficients or tree structure;
2. compute permutation importance;
3. perturb top-ranked features;
4. compare local explanations for similar samples.

## Engineering lesson
Interpretability can reveal leakage, unstable features and shortcut learning.
