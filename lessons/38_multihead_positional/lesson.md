# Lesson 38: Multi-head attention and positional information

## Multi-head attention
Instead of one attention operation, split representation space into multiple heads. Each head can learn different interaction patterns.

## Why position is needed
Self-attention by itself does not inherently know whether a token was first or tenth. Transformers therefore inject position information.

Learn:
- sinusoidal positional encoding
- learned positional embeddings
- relative position concepts

## Shape practice
For:
- batch B
- sequence T
- model dimension D
- heads H

head dimension is usually D/H.

Track shapes through:
X -> Q,K,V -> split heads -> attention -> concatenate -> output projection.

## Exercise
Implement head splitting and recombination in PyTorch without using `nn.MultiheadAttention`.
