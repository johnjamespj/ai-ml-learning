"""Execute canonical notebooks in independent kernels; errors are never passes.

Source notebooks are not rewritten. Executed copies, tracebacks and JSON results
are stored under artifacts/. External/hardware paths must be declared explicitly.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]


def source_paths(group: str) -> list[Path]:
    groups = {
        'lessons': sorted((ROOT / 'lessons').glob('*/lesson.ipynb')),
        'math': sorted((ROOT / 'math').glob('*.ipynb')),
        'papers': sorted((ROOT / 'papers/notebooks').glob('*.ipynb')),
        'extras': sorted((ROOT / 'notebooks').glob('*.ipynb')),
        'capstone': sorted((ROOT / 'capstones/rf_detection').glob('*.ipynb')),
    }
    if group == 'smoke':
        choices = [('lessons', '00_'), ('lessons', '01_'), ('lessons', '07_'),
                   ('lessons', '21_'), ('lessons', '26_'), ('lessons', '37_'),
                   ('lessons', '54_'), ('lessons', '58_'),
                   ('math', '02_'), ('math', '11_'),
                   ('papers', '01_'), ('papers', '17_')]
        return [p for g, prefix in choices for p in groups[g]
                if (p.parent.name if g == 'lessons' else p.name).startswith(prefix)]
    if group == 'all':
        return sorted(p for values in groups.values() for p in values)
    return groups[group]


def run_one(path: Path, output: Path, timeout: int, kernel: str, baseline: bool) -> dict:
    import nbformat
    from nbclient import NotebookClient
    relative = path.relative_to(ROOT).as_posix()
    result = {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'status': 'not_run', 'seconds': 0.0, 'optional_paths': []}
    # The baseline contains a shell cell that launches Jupyter and a model download.
    # Do not execute them as an incidental effect of the first audit.
    if baseline and path.parent.name in {'00_setup', '42_huggingface_transformers'}:
        result.update(status='not_run_safety', reason='Baseline launches server/install or downloads weights')
        return result
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=nbformat.warnings.MissingIDFieldWarning)
        notebook = nbformat.read(path, as_version=4)
    result['optional_paths'] = notebook.metadata.get('course', {}).get('optional_paths', [])
    target = output / 'executed' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='course-notebook-') as work:
        try:
            # No injected variables or repaired cells here. The actual source must run.
            client = NotebookClient(notebook, timeout=timeout, kernel_name=kernel,
                                    allow_errors=False, force_raise_errors=True,
                                    resources={'metadata': {'path': work}})
            client.execute()
            result['status'] = ('cpu_passed_optional_not_run'
                                if result['optional_paths'] else 'passed')
        except Exception as exc:
            result.update(status='failed', error=type(exc).__name__, detail=str(exc))
        finally:
            result['seconds'] = round(time.perf_counter() - started, 3)
            nbformat.write(notebook, target)
    print(f"{result['status']:29} {relative} ({result['seconds']}s)", flush=True)
    return result


def source_manifest() -> dict:
    paths=[]
    for folder in ('coursekit','capstones','labs','lessons','math','papers/notebooks','notebooks','tools','assessment','instructor','requirements'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.ipynb','.txt'} and '__pycache__' not in p.parts:
                paths.append(p)
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['all','smoke','lessons','math','papers','extras','capstone'], default='smoke')
    parser.add_argument('--include', default='', help='Filter selected paths by substring')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/notebooks')
    parser.add_argument('--timeout', type=int, default=180, help='Per-cell seconds; never unlimited')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--kernel', default='python3')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error('workers and timeout must be positive')
    paths = [p for p in source_paths(args.group) if args.include in str(p)]
    if not paths:
        parser.error('No notebooks selected')
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ.update(MPLBACKEND='Agg', COURSE_ROOT=str(ROOT), COURSE_RESULTS_DIR=str(args.output/'runs'),
                      COURSE_RUN_DOWNLOADS='0', COURSE_RUN_DOCKER='0', COURSE_RUN_GPU='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH','')
    manifest=source_manifest()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one,p,args.output,args.timeout,args.kernel,args.baseline) for p in paths]
        for future in as_completed(futures):
            results.append(future.result())
            (args.output/'results.partial.json').write_text(json.dumps(results, indent=2))
    packages = {}
    for pkg in ['numpy','scipy','scikit-learn','torch','nbclient','nbformat','ipykernel','fastapi']:
        try: packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: packages[pkg] = None
    counts = {s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}
    report = {'scope':args.group,'baseline':args.baseline,'source_count':len(paths),
              'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':packages},
              'counts':counts,'results':sorted(results,key=lambda r:r['path']),
              'source_manifest':manifest,'source_changed_during_run':manifest!=source_manifest(),
              'limitation':'CPU execution is not evidence of CUDA, model downloads, Docker runtime or scientific reproduction.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(counts, indent=2), flush=True)
    return int(any(r['status']=='failed' for r in results) or report['source_changed_during_run'])

if __name__ == '__main__':
    raise SystemExit(main())
