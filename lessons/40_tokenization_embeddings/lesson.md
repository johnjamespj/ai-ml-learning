# Lesson 40: Tokenization and embeddings

Language models operate on token IDs, not raw words.

## Tokenization
Modern tokenizers usually represent text using subword units rather than one token per word or character.

Pipeline:

text -> tokenizer -> token IDs -> embeddings -> model

## Embedding layer
```python
import torch
from torch import nn

vocab_size = 10000
d_model = 128

embedding = nn.Embedding(vocab_size, d_model)
ids = torch.tensor([[12, 91, 4, 700]])
x = embedding(ids)

print(x.shape)
```

Embeddings are learned vector representations.

## Similarity
Cosine similarity is commonly used to compare vector direction.

```python
import torch.nn.functional as F
similarity = F.cosine_similarity(x[:,0,:], x[:,1,:])
```

## Important
Embedding similarity means similarity in the learned representation space. It is not a guarantee of truth, causality or human-equivalent meaning.
