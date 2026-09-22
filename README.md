# AI/ML Learning Lab

A hands-on path from Python scientific computing to modern AI systems.

This repository contains **78 ordered lessons (00-77)** plus runnable labs, automated tests, reference solutions, notebooks, capstone projects, and a landmark-paper reproduction track.

## Start here

1. Read [STUDY_PLAN.md](STUDY_PLAN.md).
2. Use [CURRICULUM.md](CURRICULUM.md) as the canonical lesson index.
3. Use [PYTHON_TOOLBOX.md](PYTHON_TOOLBOX.md) to map concepts to Python libraries.
4. Begin [Lesson 00](lessons/00_setup/lesson.md).
5. Track implementation work in [LAB_PROGRESS.md](LAB_PROGRESS.md).

## Learn by building

The course has four intertwined layers:

### 1. Lessons
Concept, intuition, math, failure modes, and library usage.

### 2. Runnable labs
Open [labs/README.md](labs/README.md).

Labs contain:
- `starter.py`
- `solution.py`
- `test_lab.py`
- experiments
- explanation prompts

Example:
```bash
LAB_TARGET=starter pytest labs/lab01_linear_regression/test_lab.py -q
```

### 3. Notebooks
Open [notebooks/README.md](notebooks/README.md).

Notebooks are for visualization and experimentation. Reusable logic belongs in Python modules.

### 4. Landmark papers
Open [papers/README.md](papers/README.md) and [papers/PAPER_TRACK.md](papers/PAPER_TRACK.md).

Each paper activity asks you to:
- understand the pre-paper problem;
- state the central claim;
- reconstruct the key mechanism;
- reproduce one trend at honest educational scale;
- perform an ablation;
- explain the evidence;
- defend the paper without notes.

This means you do not merely read *Attention Is All You Need*. You implement attention, test masks and shapes, alter the scaling term, train a toy Transformer, and explain what actually changed.

## Major tracks

| Track | Topics | Main tools |
|---|---|---|
| Foundations | Python, data, linear algebra, calculus, probability | NumPy, pandas, SciPy |
| Classical ML | regression, classification, trees, boosting, SVM, clustering, PCA | scikit-learn |
| Neural nets | forward pass, backprop, losses, optimizers | NumPy |
| Deep learning | autograd, training loops, CNNs | PyTorch |
| Sequences | RNN, LSTM, GRU | PyTorch |
| Transformers | attention, multi-head attention, tokenization | PyTorch |
| LLM engineering | Hugging Face, LoRA, embeddings, RAG | Transformers, PEFT |
| Generative AI | autoencoders, VAEs, GANs, diffusion | PyTorch |
| Reinforcement learning | Q-learning, DQN, actor-critic concepts | Gymnasium, PyTorch |
| Signals/Radar | features, spectrograms, detection, domain shift | SciPy, scikit-learn, PyTorch |
| Production | tracking, APIs, Docker, monitoring, GPU/distributed training | FastAPI, Docker, PyTorch |
| Advanced ML | Bayesian ML, uncertainty, causality, GNNs, multimodal, interpretability | mixed |
| Systems | agents, robustness, data-centric AI, ML system design | full stack |

## Capstone philosophy

Every sophisticated model should answer to a simpler baseline. Every metric should match the actual task. Every experiment should preserve provenance. Every deployment should include a plan for detecting when reality changes underneath the model.

## Learning rule

If you cannot explain the inputs, outputs, parameters, objective/loss, evaluation metric, assumptions, and failure modes, the model is still a black box. Open the box.
