# Landmark Paper Track

Read these alongside the matching course lessons.

| # | Paper | Year | Reproduction target |
|---|---|---:|---|
| 01 | Rosenblatt, *The Perceptron* | 1958 | perceptron learning on separable vs XOR data |
| 02 | Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors* | 1986 | backprop on a tiny hidden-layer network |
| 03 | LeCun et al., *Gradient-Based Learning Applied to Document Recognition* | 1998 | LeNet-style CNN on MNIST/FashionMNIST |
| 04 | Krizhevsky, Sutskever & Hinton, *ImageNet Classification with Deep CNNs* | 2012 | AlexNet ideas: ReLU, augmentation, dropout |
| 05 | Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* | 2013 | skip-gram/CBOW embeddings on a tiny corpus |
| 06 | Kingma & Welling, *Auto-Encoding Variational Bayes* | 2013 | VAE reparameterization + latent interpolation |
| 07 | Goodfellow et al., *Generative Adversarial Networks* | 2014 | GAN on a 2-D distribution |
| 08 | Mnih et al., *Human-level control through deep reinforcement learning* | 2015 | DQN ingredients on a small Gymnasium task |
| 09 | He et al., *Deep Residual Learning for Image Recognition* | 2015 | plain vs residual network at increasing depth |
| 10 | Vaswani et al., *Attention Is All You Need* | 2017 | scaled dot-product + multi-head attention |
| 11 | Devlin et al., *BERT* | 2018 | masked-language-model adaptation at small scale |
| 12 | Brown et al., *Language Models are Few-Shot Learners* | 2020 | in-context prompting behavior, not full pretraining |
| 13 | Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* | 2020 | SimCLR-style contrastive embeddings |
| 14 | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | 2020 | retriever + generator with retrieval metrics |
| 15 | Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* | 2020 | toy forward-noise/reverse-denoise process |
| 16 | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* | 2021 | CLIP-style aligned toy embeddings |
| 17 | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* | 2021 | full vs low-rank trainable update |
| 18 | Ronneberger et al., *U-Net* | 2015 | skip-connected encoder-decoder segmentation |
| 19 | Ioffe & Szegedy, *Batch Normalization* | 2015 | training with/without batch normalization |
| 20 | Srivastava et al., *Dropout* | 2014 | generalization with/without dropout |
| 21 | Kingma & Ba, *Adam* | 2014 | optimizer trajectories on same objective |

## Stable links for several core papers
- VAE: https://arxiv.org/abs/1312.6114
- GAN: https://arxiv.org/abs/1406.2661
- ResNet: https://arxiv.org/abs/1512.03385
- Transformer: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805
- RAG: https://arxiv.org/abs/2005.11401
- DDPM: https://arxiv.org/abs/2006.11239
- LoRA: https://arxiv.org/abs/2106.09685

## Rule
For giant-scale papers, reproduce the **mechanism and claimed trend**, not the original compute budget. State the scale gap explicitly.
