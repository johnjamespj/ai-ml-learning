# Reproducibility and evidence export

Every paper notebook now creates a unique results directory, records its source hash and available Git commit, Python/package versions, declared configuration and scope, captures figures, and exports numeric diagnostics. `config.json`, `metrics.csv`, `ablation.csv`, `diagnostics.json`, `conclusions.md` and `run_status.json` separate measured output from unfinished interpretation.

`diagnostics.json` is not a validated benchmark table. Explicit scientific metrics should be logged with their meaning:

```python
experiment.log('baseline', seed=0, metrics={'validation_accuracy': 0.0})
```

Replace the example value with your actual measured variable. Never type a desired score. For paired-seed experiments:

```python
from coursekit.experiments import run_trials
run_trials(experiment, {'baseline': run_baseline, 'without_component': run_ablation}, seeds=(0,1,2))
```

Each function accepts a seed and returns real named scalar metrics. Exceptions and non-finite values fail visibly. A scaffold marked `not_run` remains incomplete. Running a notebook does not mean every written extension or ablation has been performed.

Before claiming reproduction, predeclare the source paper, target claim, exact dataset/splits, scale gap, baseline, metric, seed list, compute budget and selection procedure. Keep validation distinct from final testing. Record failed runs and negative outcomes. Do not reuse an attractive single-seed curve as evidence of a general result.

The source repairs also correct several mechanism examples: ResNet plain/residual blocks now have matching linear-layer counts; the Transformer experiment has balanced labels and a held-out final test; DDPM uses a terminal noise schedule close to its sampling prior; SimCLR includes the full 2N contrastive denominator. These changes make the experiments more defensible but do not recreate the original large-scale benchmarks.
