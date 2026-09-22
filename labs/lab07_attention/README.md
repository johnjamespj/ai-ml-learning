# Lab 07: Attention Is All You Need, Core Mechanism

Paper companion: Transformer activity.

Implement scaled dot-product attention from raw PyTorch tensor operations.

Requirements:
- QK^T scores
- divide by sqrt(d_k)
- optional mask
- softmax
- weighted sum of V

Run:
```bash
LAB_TARGET=starter pytest labs/lab07_attention/test_lab.py -q
```

Explain the dimensions of every tensor.
