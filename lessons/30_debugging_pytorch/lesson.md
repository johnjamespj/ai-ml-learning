# Lesson 30: Debugging PyTorch models

Most deep-learning bugs are not solved by "train longer."

## Debugging ladder
1. Verify input shapes.
2. Verify target shapes/dtypes.
3. Inspect min/max/mean/std.
4. Run one forward pass.
5. Inspect loss before training.
6. Overfit a tiny batch.
7. Inspect gradient magnitudes.
8. Check for NaN/Inf.
9. Verify train/eval mode.
10. Confirm data split and labels.

## Useful checks
```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm().item())
```

## Common mistakes
- wrong output dimension
- applying softmax before CrossEntropyLoss
- forgetting optimizer.zero_grad()
- target dtype is float instead of long for class indices
- model on GPU but data on CPU
- evaluation with dropout still active
- data leakage
- normalization computed using test data

## Exercise
Create three deliberately broken training scripts, one each for shape, gradient and data problems. Diagnose them systematically.
