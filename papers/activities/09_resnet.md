# Paper Activity 09: ResNet

**Paper:** He et al., "Deep Residual Learning for Image Recognition" (2015), https://arxiv.org/abs/1512.03385

## Central claim
Residual connections make substantially deeper networks easier to optimize.

## Reproduce
Train plain and residual networks of increasing depth on a small image or synthetic classification task.

Plot:
- training loss
- validation accuracy
- gradient norms

## Ablation
Replace residual addition with no skip path while keeping comparable blocks.

## Explain
Why is learning F(x)=H(x)-x potentially easier than directly learning H(x)?
