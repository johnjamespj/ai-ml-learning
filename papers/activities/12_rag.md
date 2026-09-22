# Paper Activity 12: Retrieval-Augmented Generation

**Paper:** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020), https://arxiv.org/abs/2005.11401

## Reproduce the system idea
Use this repository as a document corpus.

Build:
query -> embeddings -> retrieval -> context -> generator

## Required measurement
Create labeled question/relevant-document pairs and report:
- Recall@1
- Recall@5
- MRR

Then evaluate final answers separately.

## Ablations
- no retrieval
- random retrieval
- top-1 vs top-k
- different chunk sizes

## Explain
Why can answer quality improve even if the generator parameters never learn the retrieved knowledge?
