# Study Plan

Treat this repository as an engineering apprenticeship, not a reading list.

## Normal lesson cycle

For each lesson:
1. Read the concept and math.
2. Predict outputs/shapes before executing examples.
3. Type and modify the code.
4. Complete the matching runnable lab when one exists.
5. Make the lab tests pass against `starter.py`.
6. Run at least one experiment that changes an assumption.
7. Explain the result from memory.
8. Commit your work.

## Paper cycle

Every few lessons, complete a matching landmark-paper activity.

Use:
- [papers/PAPER_TRACK.md](papers/PAPER_TRACK.md)
- [papers/REPRODUCTION_TEMPLATE.md](papers/REPRODUCTION_TEMPLATE.md)
- [papers/PAPER_CLUB.md](papers/PAPER_CLUB.md)

A paper is complete when you can explain the world before it, its actual novelty, the evidence, an ablation, its scale limitations, and what survived into modern practice.

## Stages

### Stage 1: Foundations
Lessons 00-07.

Labs: 01.

Paper: Perceptron.

Gate: Phase 1 checkpoint.

### Stage 2: Classical ML
Lessons 08-18.

Labs: 02-03.

Gate: Classical ML benchmark.

### Stage 3: Neural networks from first principles
Lessons 19-24.

Lab: 04.

Paper: Backpropagation.

Gate: NumPy neural network project.

### Stage 4: PyTorch and computer vision
Lessons 25-34.

Labs: 05, 06, 09, 10.

Papers: LeNet, AlexNet, ResNet.

Gate: PyTorch classifier + spectrogram CNN.

### Stage 5: Sequences, Transformers and LLMs
Lessons 35-45.

Labs: 07, 12.

Papers: word2vec, Transformer, BERT, RAG, LoRA.

Gate: Mini Transformer + measured RAG project.

### Stage 6: Generative AI and RL
Lessons 46-51.

Labs: 08, 11.

Papers: VAE, GAN, DDPM, DQN.

Gate: generative-model and RL projects.

### Stage 7: Signals, deployment and advanced ML
Lessons 52-77.

Lab: 10 plus the deployment/capstone projects.

Paper extensions: contrastive learning / CLIP-style multimodal representation learning.

Gate: deployed model + final capstone.

## Mastery rule

You know a topic when you can:
- explain the problem it solves;
- derive or describe the mechanism;
- implement a simplified version;
- use the standard Python tool;
- choose an evaluation method;
- identify failure modes;
- compare against a simpler baseline;
- explain a landmark paper that shaped the technique.

Running a notebook successfully is evidence of execution, not evidence of understanding.
