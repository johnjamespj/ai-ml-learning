# Lesson 42: Hugging Face Transformers

Once you understand tokenization, embeddings, attention and language modeling, use higher-level libraries productively.

## Tokenizer and model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(
    name,
    num_labels=2,
)

batch = tokenizer(
    ["signal detected", "noise only"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

logits = model(**batch).logits
print(logits.shape)
```

## Learn the abstractions
Understand:
- model hub
- configs
- tokenizer
- pretrained weights
- task-specific heads
- datasets
- batching/padding
- attention masks

## Rule
Never let a convenience API replace understanding of the tensor shapes and task objective underneath it.
