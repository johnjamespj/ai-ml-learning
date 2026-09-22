# Paper Club Workflow

Use one paper cycle every 1-2 weeks while progressing through the normal lessons.

## Session A: read
Before reading, write:
- what problem you think the title addresses;
- what baseline you expect;
- what result would convince you.

Then read abstract, introduction, figures/tables, method, experiments, limitations/conclusion.

## Session B: reconstruct
Without looking at code from the authors:
- derive the key equation;
- draw the model;
- implement the smallest central mechanism;
- write tests for shape/math invariants.

## Session C: reproduce
Reproduce one meaningful trend or ablation. Do not chase the paper's exact headline number unless the compute/data scale is actually comparable.

## Session D: defend
Give a five-minute explanation:
1. problem before the paper;
2. central novelty;
3. strongest evidence;
4. weakest assumption;
5. what survived into modern systems;
6. what you would test next.

## Required artifacts
For each paper produce:
- `notes.md`
- `reproduce.py` or notebook
- one plot/table
- `results.md`
- reproducibility config/seeds
- a paragraph called **Scale gap** stating how your setup differs from the paper
