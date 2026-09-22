# Lesson 62: Distributed training

## Why distribute
Large models/datasets can exceed one device or one machine's practical throughput.

## Data parallelism
Replicate the model on multiple workers, split batches and synchronize gradients.

## Concepts
- process
- rank
- world size
- collective operations
- all-reduce
- DistributedDataParallel
- distributed sampler

## Beyond data parallelism
Learn the purpose of:
- model parallelism
- tensor parallelism
- pipeline parallelism
- sharding/FSDP concepts

## Important
Distributed training adds communication and coordination overhead. Two GPUs do not automatically mean 2x speed.

## Exercise
Run a tiny multi-process DistributedDataParallel example on available hardware or study the execution flow if only one device is available.
