# Lab 06: Residual Connections

Paper companion: ResNet activity.

Implement a residual MLP block and compare a deep plain MLP with a deep residual MLP on a synthetic task.

Run:
```bash
LAB_TARGET=starter pytest labs/lab06_residual_network/test_lab.py -q
```

Then train both at several depths and plot training loss and gradient norms.
