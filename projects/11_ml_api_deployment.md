# Project 11: Deploy an ML Model

## Mission
Take a trained model from notebook-land into a testable service.

## Build
Create:
- FastAPI service
- request/response schemas
- preprocessing pipeline
- `/health`
- `/predict`
- model version metadata
- unit/integration tests
- Dockerfile
- reproducible configuration

## Measure
Record:
- startup time
- p50/p95 latency
- throughput
- batch-size effect
- memory footprint
- invalid-request behavior

## Correctness gate
The same fixed sample must produce equivalent predictions in:
1. offline evaluation code;
2. the FastAPI service;
3. the Docker container.

## Stretch
Add structured logs, simple drift statistics and a benchmark script.
