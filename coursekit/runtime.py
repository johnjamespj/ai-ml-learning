"""Explicit, CPU-friendly notebook setup. No installs, downloads, or servers."""
from __future__ import annotations
import os
from pathlib import Path
import random


def configure(seed: int = 0) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    # Set before importing torch when possible. A seed is not a cross-device guarantee.
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def root() -> Path:
    candidate = os.environ.get('COURSE_ROOT')
    if candidate:
        return Path(candidate).resolve()
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path/'coursekit').is_dir() and (path/'lessons').is_dir():
            return path
    raise RuntimeError('Open this notebook inside the cloned repository, or set COURSE_ROOT.')
