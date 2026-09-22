# Lab 12: Retrieval Before RAG

Linked lessons: 44-45. Paper companion: RAG.

Do not begin with an LLM. First prove retrieval works.

## Task
Use TF-IDF as a transparent baseline:
- fit document vectors
- retrieve top-k
- calculate Recall@k
- calculate MRR

## Tests
```bash
LAB_TARGET=starter pytest labs/lab12_retrieval_eval/test_lab.py -q
```

## Extension
Replace TF-IDF with an embedding model and compare on the exact same labeled queries.

## Explain
Give an example where a fluent generator cannot repair a retrieval failure because the needed evidence never entered context.
