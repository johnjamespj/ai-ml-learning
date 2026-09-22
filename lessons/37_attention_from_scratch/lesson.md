# Lesson 37: Attention from scratch

Attention allows each position to retrieve information from other positions.

## Query, key and value
For input representation X:

Q = XW_Q
K = XW_K
V = XW_V

Scaled dot-product attention:

Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

## NumPy implementation
```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
```

## Interpret the matrices
If sequence length is T, attention scores are T x T. Each row says how strongly one query position attends to every key position.

## Exercise
Create four tiny token vectors and hand-compute one row of attention scores. Then verify with NumPy.

## Important
Attention weights can be useful diagnostic signals, but should not automatically be treated as causal explanations.
