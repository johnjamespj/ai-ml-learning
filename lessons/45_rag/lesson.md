# Lesson 45: Retrieval-Augmented Generation

RAG combines retrieval with generation.

## Basic pipeline
1. ingest documents
2. clean/parse
3. chunk
4. embed
5. index
6. embed query
7. retrieve candidates
8. optionally rerank
9. construct context
10. generate answer
11. retain citations/evidence

## Chunking
Chunk size and overlap influence retrieval behavior. Too small can lose context; too large can dilute matching and consume context window.

## Failure modes
- relevant chunk never indexed
- embedding mismatch
- poor query formulation
- wrong chunk retrieved
- correct chunk retrieved but ignored
- answer unsupported by retrieved evidence
- stale corpus
- context overflow

## Evaluation
Separate:
1. retrieval quality;
2. groundedness;
3. task answer quality;
4. latency/cost.

## Exercise
Build a RAG system over this repository's Markdown lessons. Ask factual questions whose source lesson is known, then calculate retrieval success before judging generation.
