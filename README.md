# AI/ML Learning Lab

A hands-on path from Python scientific computing to modern AI systems.

This repository now contains **78 ordered lessons (00-77)**, checkpoints, and progressive projects spanning classical ML, deep learning, LLMs, generative AI, reinforcement learning, signal/radar ML, MLOps, and advanced ML topics.

## Start here

1. Read [STUDY_PLAN.md](STUDY_PLAN.md).
2. Use [CURRICULUM.md](CURRICULUM.md) as the canonical lesson index.
3. Keep [PYTHON_TOOLBOX.md](PYTHON_TOOLBOX.md) nearby to see which Python libraries map to each concept.
4. Begin with [lessons/00_setup/lesson.md](lessons/00_setup/lesson.md).

## How to use this repo

For every lesson:
1. Learn the intuition.
2. Work through the math.
3. Implement the core idea from scratch where useful.
4. Use the standard Python library.
5. Run an experiment.
6. Complete the exercise.
7. Explain the failure modes.
8. Build something.

Do not rush to frameworks. The goal is to understand what the library is doing for you.

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

Every sophisticated model should answer to a simpler baseline. Every metric should match the real task. Every experiment should preserve provenance. Every deployment should include a plan for detecting when reality changes underneath the model.

## Learning rule

If you cannot explain the inputs, outputs, parameters, objective/loss, evaluation metric, assumptions, and failure modes, the model is still a black box. Open the box.
