# Complete AI/ML Roadmap

## Phase 0: Environment and scientific Python
Python environments, packages, Jupyter, Git, NumPy arrays, vectorization, broadcasting, pandas, plotting, SciPy.

## Phase 1: Mathematical foundations
Vectors and matrices; dot products; matrix multiplication; norms; projections; eigenvalues/eigenvectors; derivatives; partial derivatives; gradients; chain rule; probability; random variables; distributions; expectation/variance; Bayes rule; statistics; maximum likelihood; optimization and gradient descent.

## Phase 2: Data
Data types, cleaning, missing data, encoding, normalization/standardization, leakage, train/validation/test splits, exploratory analysis, feature engineering, dimensionality.

## Phase 3: Supervised machine learning
Linear regression from scratch; loss functions; gradient descent; logistic regression; k-NN; Naive Bayes; decision trees; random forests; bagging; boosting; SVMs; kernels; regularization; hyperparameters.

Learn bias vs variance, under/overfitting, cross-validation, confusion matrices, precision, recall, F1, ROC/AUC, calibration, regression metrics and imbalanced datasets.

## Phase 4: Unsupervised learning
k-means, hierarchical clustering, DBSCAN, PCA, dimensionality reduction, anomaly detection and representation learning.

## Phase 5: Neural networks from first principles
Perceptrons, layers, activations, loss functions, forward propagation, backpropagation, computational graphs, initialization, optimizers, SGD, momentum, Adam, normalization, dropout and regularization.

Build a small neural network using only NumPy before using PyTorch.

## Phase 6: PyTorch
Tensors, autograd, Dataset/DataLoader, nn.Module, training loops, GPU training, checkpoints, experiment design and debugging.

## Phase 7: Computer vision
Image tensors, convolutions, kernels, pooling, CNNs, augmentation, transfer learning, classification, detection and segmentation concepts.

## Phase 8: Sequential data
Sequence representation, RNNs, vanishing gradients, LSTMs, GRUs, temporal prediction and time-series modeling.

## Phase 9: Attention and Transformers
Queries/keys/values, scaled dot-product attention, multi-head attention, positional information, encoder/decoder architecture, masking, tokenization and Transformer training.

Implement attention manually before relying on a framework implementation.

## Phase 10: LLMs
Language modeling, next-token prediction, tokenizers, embeddings, pretraining, instruction tuning, decoding, context windows, inference, quantization, evaluation, hallucination and safety concepts.

Tools: PyTorch and Hugging Face Transformers.

## Phase 11: Applied LLM engineering
Embedding models, semantic search, vector indexes, chunking, retrieval, RAG, reranking, tool use, structured output, fine-tuning, LoRA/PEFT and evaluation.

## Phase 12: Generative modeling
Autoencoders, VAEs, GAN concepts, diffusion models, latent representations and conditional generation.

## Phase 13: Reinforcement learning
Agent/environment/reward, Markov decision processes, value functions, Bellman equations, Q-learning, policy methods, actor-critic concepts and deep RL.

## Phase 14: Signal-processing ML
Sampling and features, FFT/STFT features, spectrograms, filtering + ML pipelines, time-frequency classification, anomaly detection, 1-D CNNs, sequence models, detection/classification tradeoffs and synthetic-data experiments.

### Radar-focused projects
- RF/spectral signal classifier
- Interference/anomaly detector
- Spectrogram CNN
- Synthetic target/detection experiment
- Time-frequency embedding explorer

## Phase 15: Production ML
Reproducibility, configuration, experiment tracking, tests, model serialization, inference APIs, FastAPI, Docker, batching, CPU/GPU inference, monitoring, drift and retraining.

## Capstones
1. Classical ML project from raw data to evaluation.
2. Neural network built from scratch.
3. PyTorch vision or signal classifier.
4. Transformer/attention implementation.
5. RAG application with measured retrieval quality.
6. End-to-end signal/radar ML system.
7. Deploy one model as a tested API.

## Mastery test
For each major model answer:
- What problem does it solve?
- What assumptions does it make?
- What is learned?
- What objective is optimized?
- How does training work?
- How can it fail?
- Which metric should evaluate it?
- What simpler baseline should it beat?
