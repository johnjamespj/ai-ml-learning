# Lesson 77: ML system design

A production ML system is more than a model.

## End-to-end architecture
data sources
-> ingestion
-> validation
-> storage/versioning
-> feature/preprocessing pipeline
-> training
-> evaluation
-> model registry/artifact
-> serving
-> monitoring
-> feedback/retraining

## Design questions
- batch or real-time inference?
- latency budget?
- throughput?
- CPU/GPU?
- how fresh must predictions be?
- when do labels arrive?
- what happens on failure?
- how is the model version rolled back?

## Exercise
Design two architectures:
1. offline radar-data analysis;
2. low-latency streaming signal classification.

For each, draw data flow, failure points, monitoring and retraining triggers.

## Final perspective
Model choice matters, but system boundaries, data quality and evaluation design often decide whether the project succeeds.
