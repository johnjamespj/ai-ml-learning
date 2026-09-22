# Project 07: Embedding Search and RAG

## Corpus
Use the Markdown files in this repository as the first corpus.

## Part A: retrieval
Implement:
- parser
- chunker
- embedding generation
- vector storage/index
- query embedding
- top-k retrieval

Create at least 30 question/relevant-document pairs and calculate Recall@k and MRR.

## Part B: generation
Pass retrieved evidence into a generator.

Require the system to return:
- answer
- source filenames
- retrieved evidence identifiers

## Part C: experiments
Compare:
- chunk sizes
- overlap
- top-k
- embedding models
- optional reranking

## Main lesson
A convincing demo is not an evaluation. Measure the retrieval subsystem separately.
