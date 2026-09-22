"""Bound CPU resource use and make repository modules importable in tests."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass
