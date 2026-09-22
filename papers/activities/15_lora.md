# Paper Activity 15: LoRA

**Paper:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021), https://arxiv.org/abs/2106.09685

## Reproduce the mechanism
Given a frozen linear layer W, learn:

W' = W + B A

where rank r is much smaller than the original dimensions.

## Compare
- full fine-tuning
- frozen base + classifier/head only
- LoRA at ranks 1, 2, 4, 8

Report:
- trainable parameters
- memory estimate
- validation quality
- training time

## Explain
Why can a low-rank update be expressive enough for adaptation?
