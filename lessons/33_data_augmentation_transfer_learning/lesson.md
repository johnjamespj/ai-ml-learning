# Lesson 33: Data augmentation and transfer learning

## Augmentation
Training-time transformations create plausible variations of existing data.

Examples for images:
- crop
- flip
- rotation
- brightness/contrast

Augmentations must preserve the target. A transformation that changes class meaning corrupts training.

## Transfer learning
A model pretrained on a large dataset can provide useful features for a smaller target dataset.

Typical workflow:
1. load pretrained backbone;
2. replace classifier head;
3. freeze backbone initially;
4. train new head;
5. optionally unfreeze later layers and fine-tune carefully.

## Why it works
Early/mid-level representations often capture reusable visual structure.

## Signal analogy
A pretrained encoder for waveforms or spectrograms can similarly provide representations for downstream signal tasks.

## Exercise
Compare:
- small CNN trained from scratch
- pretrained vision backbone with frozen feature extractor
- partially fine-tuned model

Compare not just final accuracy, but data efficiency and training time.
