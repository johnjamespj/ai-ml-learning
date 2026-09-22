"""Record real run outputs without inventing conclusions or completed ablations."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Callable, Mapping, Any
from uuid import uuid4
from .runtime import root


def _json(value: Any) -> Any:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite values are not valid experimental evidence')
    if value is None or isinstance(value, (str,int,float,bool)):
        return value
    raise TypeError(f'Unsupported evidence type: {type(value).__name__}')


class Experiment:
    def __init__(self, name: str, config: Mapping[str, Any], source: str | None = None):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Use an alphanumeric experiment name, with hyphens/underscores')
        base = Path(os.environ.get('COURSE_RESULTS_DIR', root()/'results'))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.directory = base / name / f'{stamp}-{uuid4().hex[:8]}'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict] = []
        self.figure_count = 0
        self._show = None
        self._plt = None
        versions = {}
        for package in ('numpy','pandas','scipy','scikit-learn','torch','matplotlib','nbclient'):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package] = None
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root(),stderr=subprocess.DEVNULL,text=True).strip()
            dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root(),text=True).strip())
        except (OSError,subprocess.CalledProcessError):
            commit, dirty = None, None
        source_path = root()/source if source else None
        payload = dict(config)
        payload.update(python=platform.python_version(),platform=platform.platform(),packages=versions,
                       commit=commit,working_tree_dirty=dirty,source=source,
                       source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path and source_path.is_file() else None)
        (self.directory/'config.json').write_text(json.dumps(_json(payload),indent=2))
        (self.directory/'ablation.csv').write_text('variant,seed,metric,value,status\nstudent_ablation,,,,not_run\n')
        (self.directory/'conclusions.md').write_text('# Conclusions\n\nStatus: learner interpretation pending.\n\n## Measured result\n\n## Scale gap\n\n## Ablation and uncertainty\n\n## Failure cases\n')

    def log(self, variant: str, seed: int, metrics: Mapping[str, float]) -> None:
        for metric,value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f'Non-finite metric: {metric}')
            self.rows.append(dict(variant=variant,seed=int(seed),metric=str(metric),value=value,status='measured'))
        with (self.directory/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['variant','seed','metric','value','status'])
            writer.writeheader(); writer.writerows(self.rows)

    def capture_figures(self) -> None:
        import matplotlib.pyplot as plt
        self._plt, self._show = plt, plt.show
        def save_and_show(*args, **kwargs):
            for number in plt.get_fignums():
                self.figure_count += 1
                plt.figure(number).savefig(self.directory/f'figure_{self.figure_count:03d}.png',dpi=140,bbox_inches='tight')
            return self._show(*args, **kwargs)
        plt.show = save_and_show

    def finish(self, namespace: Mapping[str, Any] | None = None) -> Path:
        # Diagnostics are NOT silently promoted to validation metrics.
        diagnostics = {}
        for key in ('loss','losses','acc','mse','kl','recon','rows','results','variants','df','table','h','h0','h1','ious','ioun'):
            if namespace is None or key not in namespace:
                continue
            value = namespace[key]
            if hasattr(value,'to_dict'):
                value = value.to_dict(orient='records')
            try: diagnostics[key] = _json(value)
            except (TypeError,ValueError): pass
        (self.directory/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        if not self.rows:
            (self.directory/'metrics.csv').write_text('variant,seed,metric,value,status\n')
        if self._plt is not None and self._show is not None:
            self._plt.show = self._show
        (self.directory/'run_status.json').write_text(json.dumps({
            'execution':'completed','interpretation':'pending','ablation':'see ablation.csv',
            'explicit_metric_rows':len(self.rows),'captured_figures':self.figure_count},indent=2))
        return self.directory


def run_trials(experiment: Experiment, variants: Mapping[str, Callable[[int], Mapping[str,float]]], seeds=(0,1,2)) -> None:
    """Run each variant on the same seeds; exceptions and NaNs fail visibly."""
    for name, run in variants.items():
        for seed in seeds:
            experiment.log(name, seed, run(seed))
    with (experiment.directory/'ablation.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['variant','seed','metric','value','status'])
        writer.writeheader(); writer.writerows(experiment.rows)
