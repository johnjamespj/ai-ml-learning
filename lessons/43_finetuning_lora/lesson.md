# Lesson 43: Fine-tuning and LoRA

## Full fine-tuning
Update most or all pretrained model parameters for your task.

Benefits:
- maximum flexibility

Costs:
- memory
- compute
- storage
- risk of overfitting or catastrophic forgetting

## Parameter-efficient fine-tuning
LoRA introduces small low-rank trainable updates while leaving the main pretrained weights frozen.

Conceptually:

W' = W + BA

where B and A are low-rank matrices.

## Why low rank?
A rank-r update uses many fewer trainable parameters than an unrestricted full-size weight update.

## Experiment
Take a small pretrained Transformer and compare:
- frozen embeddings/features + classifier
- full fine-tuning
- LoRA/PEFT

Report trainable parameter count, memory use, training time and validation quality.

## Important
Fine-tuning does not guarantee factual correctness. Evaluation must match the intended task.
