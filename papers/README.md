# Landmark Paper Reproduction Track

This directory is the research-history companion to the 78-lesson AI/ML course.

## Start here
- [Notebook track](NOTEBOOK_TRACK.md)
- [Paper table](PAPER_TRACK.md)
- [Paper club workflow](PAPER_CLUB.md)
- [Reproduction report template](REPRODUCTION_TEMPLATE.md)

## Notebook structure

Every paper notebook asks you to move through:

**before-reading questions → paper claim → implementation → reproduction → figure/table → ablation → scale gap → defense**

The notebooks are intentionally small enough to run on normal development hardware. They reproduce **ideas and evidence patterns**, not giant original training runs.

## What "reproduce" means here

A valid educational reproduction might show:
- perceptron convergence on separable data;
- residual connections improving optimization at depth;
- positional information being necessary for order-sensitive Transformer tasks;
- retrieval improving access to external facts;
- LoRA succeeding when the required task update is low-rank.

It does **not** mean claiming the original ImageNet, Atari, web-scale language, or billion-parameter benchmark was recreated.

Run:
```bash
python tools/validate_paper_notebooks.py
```

to structurally validate the full paper-notebook track.
