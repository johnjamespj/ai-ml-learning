# Lab 05: PyTorch Training Loop

Linked lessons: 25-30.

## Mission
Implement:
- an MLP with `nn.Module`
- a reproducible synthetic dataset
- a training step
- evaluation without gradient tracking
- best-validation checkpoint logic

## Tests
```bash
LAB_TARGET=starter pytest labs/lab05_pytorch_training/test_lab.py -q
```

## Experiments
Compare SGD and Adam using the same initialization. Measure loss, validation accuracy and wall-clock time.

## Explain
Why does PyTorch accumulate gradients, and why should evaluation use `model.eval()` plus `torch.no_grad()`?
