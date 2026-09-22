# Lesson 75: Robustness and ML security

A model can fail because inputs are noisy, shifted, malformed or deliberately manipulated.

## Concepts
- adversarial examples
- data poisoning
- model extraction concepts
- prompt injection in LLM systems
- retrieval poisoning
- insecure model deserialization
- supply-chain risk

## Defensive mindset
Validate inputs, constrain trust boundaries, isolate untrusted content, track provenance, test robustness and minimize privileged actions.

## Exercise
For a simple image or tabular classifier, perturb inputs within a bounded range and map performance degradation.

For RAG, add an untrusted document containing conflicting instructions and verify that retrieved content is treated as data rather than system authority.
