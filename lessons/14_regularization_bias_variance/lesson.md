# Lesson 14: Bias, variance and regularization

## Bias vs variance
Underfitting: the model cannot represent important structure.
Overfitting: the model captures training-specific noise and fails to generalize.

Training score alone cannot diagnose generalization.

## Regularization
Regularization penalizes model complexity.

L2 adds a squared-weight penalty:
loss + lambda * sum(w²)

L1 adds:
loss + lambda * sum(abs(w))

L1 can encourage sparse coefficients. L2 tends to shrink weights smoothly.

## Experiment
Generate polynomial data. Fit polynomial models of increasing degree and plot training and validation errors. You should see the transition from underfit to useful fit to overfit.

Then apply Ridge and Lasso.

## Questions
Why does more model capacity sometimes hurt? Why does regularization introduce bias intentionally? Why should preprocessing live inside a cross-validation pipeline?
