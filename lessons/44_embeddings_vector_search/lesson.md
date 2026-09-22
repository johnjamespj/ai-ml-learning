# Lesson 44: Embeddings and vector search

## Retrieval idea
Represent documents and queries as vectors. Search for nearby vectors.

Pipeline:

documents -> chunks -> embeddings -> index

query -> embedding -> nearest neighbors -> retrieved chunks

## Similarity
Common metrics:
- cosine similarity
- dot product
- Euclidean distance

## Tiny NumPy search
```python
import numpy as np

docs = np.random.randn(100, 64)
query = np.random.randn(64)

docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
query = query / np.linalg.norm(query)

scores = docs @ query
top = np.argsort(scores)[::-1][:5]
print(top, scores[top])
```

## Retrieval quality
Do not evaluate a RAG system only by reading final answers.

Measure retrieval directly:
- Recall@k
- Precision@k
- MRR
- nDCG when ranking relevance is graded

## Exercise
Build a small semantic-search system over your own lesson notes.
