# Paper Activity 11: BERT

**Paper:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018), https://arxiv.org/abs/1810.04805

## Educational reproduction
Do not pretrain BERT from scratch.

Instead:
1. inspect tokenization and attention masks;
2. use a small pretrained encoder;
3. fine-tune on a small classification dataset;
4. compare frozen encoder vs full/partial fine-tuning.

## Explain
What does bidirectional masked-language modeling permit that a causal next-token objective does not?

## Historical analysis
Identify which original pretraining objectives remained influential and which were deemphasized in later models.
