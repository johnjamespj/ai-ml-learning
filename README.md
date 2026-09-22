# AI/ML Learning Lab

A notebook-first, hands-on path from Python scientific computing to modern AI systems.

The course contains **78 Jupyter lessons**, runnable activities, automated labs/tests, capstone projects, and landmark-paper reproductions.

## Start here

1. Install the environment from `requirements.txt`.
2. Launch Jupyter:
   ```bash
   jupyter lab
   ```
3. Open [NOTEBOOK_INDEX.md](NOTEBOOK_INDEX.md).
4. Start with [Lesson 00](lessons/00_setup/lesson.ipynb).
5. Use [STUDY_PLAN.md](STUDY_PLAN.md) for the full progression.

## Every lesson is executable

Each `lessons/<topic>/lesson.ipynb` contains:
- the lesson explanation and math;
- executable versions of the original code examples;
- a runnable activity/experiment;
- an explanation checkpoint;
- environment-safe fallbacks where practical.

The intended rhythm is:

**read → run → change → measure → explain**

## Labs

The `labs/` directory adds starter-code exercises, reference solutions, and pytest checks.

Example:
```bash
LAB_TARGET=starter pytest labs/lab04_numpy_mlp/test_lab.py -q
```

## Landmark papers

The `papers/` track asks you to reproduce central mechanisms and results at honest educational scale, perform ablations, and defend what the paper actually contributed.

Start at:
- [Paper track](papers/PAPER_TRACK.md)
- [Reproduction template](papers/REPRODUCTION_TEMPLATE.md)
- [Paper club workflow](papers/PAPER_CLUB.md)

## Course philosophy

Every sophisticated model should answer to a simpler baseline. Every metric should match the task. Every experiment should preserve provenance. Every notebook should end with an explanation you can defend without the notebook open.

If you cannot explain the inputs, outputs, parameters, objective, evaluation metric, assumptions, and failure modes, the model is still a black box. Open the box.
