# Paper Activity 14: Contrastive Learning, SimCLR and CLIP

**Papers:**
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020), arXiv:2002.05709.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021), arXiv:2103.00020.

## SimCLR reproduction
Create two augmented views of each image and train an encoder so positive pairs are close and other samples are farther apart.

## CLIP-style extension
Create a toy paired image/text or signal/metadata dataset and align representations across modalities.

## Explain
Why does contrastive learning depend heavily on the definition of a positive pair?

## Signal extension
Use two augmented spectrogram views as positives and test whether learned embeddings improve signal classification.
