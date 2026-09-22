# Lesson 39: Build a Transformer block

A Transformer block combines:
- multi-head self-attention
- residual connections
- normalization
- feed-forward network

## Skeleton
```python
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x
```

## Residual connections
A residual path lets a block learn a correction to its input rather than rebuilding the whole representation.

## Feed-forward layer
Attention mixes information across positions. The feed-forward network performs nonlinear feature transformation independently at each position.

## Exercise
Implement the attention component yourself from Lesson 37/38 and swap it into this block.
