# Study Plan

This is a **notebook-first and math-first** course.

## Normal lesson cycle

For each lesson:

1. Open its **Mathematical Framework** section.
2. Work through the linked math notebook(s).
3. Re-derive the main equation/update in your own Markdown cell or on paper.
4. Predict important shapes, signs, limits, or qualitative behavior.
5. Run the lesson cells.
6. Modify the runnable activity.
7. Compare the new result with the original.
8. Explain the result using the math, not only the API behavior.
9. Complete the matching lab.
10. Complete the assigned landmark-paper reproduction when scheduled.

## Math gates

Before moving past a stage, be able to derive/explain:

- **Foundations:** dot products, projections, gradients, Bayes, likelihood, entropy.
- **Classical ML:** least squares, logistic likelihood, regularization, margin, PCA eigenproblem.
- **Neural nets:** chain rule, matrix-gradient shapes, cross-entropy gradient, initialization/gradient flow.
- **Transformers:** Q/K/V projections, scaled dot-product attention, masking, positional symmetry breaking.
- **Generative AI:** ELBO/KL, adversarial objective, forward diffusion marginal.
- **RL:** return, Bellman equations, TD error, Q-learning update.
- **Radar/signal ML:** likelihood-ratio tests, matched filtering, ROC/Pd/Pfa, covariance/whitening.
- **Uncertainty:** posterior predictive reasoning and calibration.

Finish [Math 16 · Synthesis and Derivation Workshop](math/16_math_synthesis.ipynb) before the final capstone.

## Stage gates

- Lessons 00-07 → Phase 1 checkpoint
- Lessons 08-18 → classical benchmark
- Lessons 19-24 → NumPy MLP + backprop paper
- Lessons 25-34 → PyTorch classifier + spectrogram CNN
- Lessons 35-45 → mini Transformer + RAG
- Lessons 46-51 → generative + RL projects
- Lessons 52-77 → radar/signal specialization, deployment, advanced ML, final capstone

## Mastery

A topic is complete only when you can:
- derive or justify its core equation;
- implement a simplified version;
- use the standard Python tool;
- choose a valid evaluation method;
- identify assumptions and failure modes;
- compare it with a simpler baseline;
- connect it to the relevant landmark paper.
