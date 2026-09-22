# AI/ML Learning Lab

A notebook-first, math-first, hands-on path from Python scientific computing to modern AI systems.

The course contains **78 Jupyter lessons**, **17 mathematical-framework notebooks**, runnable labs/tests, capstone projects, and **21 landmark-paper reproduction notebooks**.

## Start here

1. Install the environment from `requirements.txt`.
2. Launch Jupyter with `jupyter lab`.
3. Read [MATH_FRAMEWORK.md](MATH_FRAMEWORK.md).
4. Open [NOTEBOOK_INDEX.md](NOTEBOOK_INDEX.md).
5. Start with [Lesson 00](lessons/00_setup/lesson.ipynb) and its linked math companions.
6. Use [STUDY_PLAN.md](STUDY_PLAN.md) for the full progression.

## The course has four intertwined layers

### Mathematical framework
[Math notebooks](math/README.md) derive the linear algebra, calculus, probability, statistics, information theory, optimization, neural-network, Transformer, generative, RL, and signal-detection mathematics behind the code.

### Executable lessons
Every lesson contains a visible **Mathematical Framework** section linking to the relevant derivations, followed by executable examples and activities.

### Implementation labs
The `labs/` directory makes you implement core mechanisms and pass tests.

### Landmark papers
The `papers/` notebooks reproduce central mechanisms/trends at honest educational scale. Every paper notebook also links directly to its mathematical prerequisites.

## Required learning rhythm

**derive → predict → run → change → measure → explain → reproduce**

For every method, answer:

1. What mathematical objects are involved?
2. What objective/probability model defines the problem?
3. Why does the update or algorithm follow from that math?
4. Which assumptions are required?
5. What fails when those assumptions break?

## Course philosophy

Every sophisticated model should answer to a simpler baseline. Every metric should match the task. Every experiment should preserve provenance. Every result should be defensible mathematically and empirically.

If you can call the library but cannot derive or explain the mechanism, the method is still a black box.
