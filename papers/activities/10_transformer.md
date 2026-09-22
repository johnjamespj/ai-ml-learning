# Paper Activity 10: Attention Is All You Need

**Paper:** Vaswani et al., "Attention Is All You Need" (2017), https://arxiv.org/abs/1706.03762

## Reproduce
1. Implement scaled dot-product attention without `nn.MultiheadAttention`.
2. Implement head split/merge.
3. Implement a causal mask.
4. Build one Transformer block.
5. Train a tiny character-level language model or sequence-copy task.

## Ablations
- remove positional information
- remove scaling by sqrt(d_k)
- use one head vs multiple heads

## Explain
Trace every tensor shape through Q, K, V, scores, softmax and output.

## Landmark question
What did eliminating recurrence change about parallelization and long-range interactions?
