# Lesson 61: GPU performance engineering

## Goal
Learn why some training jobs underuse expensive hardware.

## Bottlenecks
- small batch sizes
- slow data loading
- CPU preprocessing
- host-device transfer
- synchronization
- memory pressure
- inefficient tensor shapes
- excessive Python overhead

## Tools/concepts
- pinned memory
- DataLoader workers
- mixed precision
- gradient accumulation
- profiling
- memory measurement

## Mixed precision
```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(xb)
    loss = loss_fn(logits, yb)
```

Use the appropriate scaler/workflow for your PyTorch version and hardware.

## Exercise
Profile a training loop before and after changing batch size, DataLoader workers and mixed precision.

## Rule
Optimize measured bottlenecks, not imagined ones.
