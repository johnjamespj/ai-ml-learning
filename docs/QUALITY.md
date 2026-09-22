# Quality contract and release checklist

The course distinguishes four kinds of evidence: structural validity, successful code execution, scientific adequacy of an experiment, and learner mastery. None automatically implies the next.

## Commands

```bash
python -m pip install -r requirements/cpu.txt
python -m ipykernel install --user --name course-ci
python tools/validate_course.py
python -m pytest -q
python tools/execute_notebooks.py --group smoke --kernel course-ci
python tools/execute_notebooks.py --group all --kernel course-ci --workers 2
```

The full worked-notebook inventory is 78 lessons, 17 math companions, 21 papers, two lab demonstrations and one capstone. The six exams and eight problem sets are intentionally ungraded starters; their syntax/schema and reference checkers are tested separately.

Each worked notebook receives a fresh kernel and a separate temporary working directory. No setup variables are injected by the runner. Source notebooks contain their own setup. Exceptions and cell timeouts fail the run. Executed copies and traces are retained, even after failure.

`passed` means the default CPU path executed. `cpu_passed_optional_not_run` names external paths such as downloads, GPU hardware or Docker operations that the CPU runner did not establish. A separate CI container test covers the deployed reference model. A CPU pass is not a CUDA benchmark or a distributed-training result.

## Baseline and repair record

The initial local audit of the 118 preexisting notebooks found 98 passes, 18 failures, and two safety holds. Sixteen failures were missing setup definitions, and the two extra lab demonstrations invoked unimplemented starters. The setup notebook also tried to install dependencies and launch Jupyter from inside a kernel. These are substantive defects, not just formatting issues.

The repair scripts edit source in a reviewable diff; execution never patches cells silently. Stable cell IDs, explicit seed setup, current numerical integration, a shifted forecasting feature and independent threshold testing were added. A later run recorded one kernel-startup infrastructure failure; preserve that attempt as well as the clean rerun.

## CI

`.github/workflows/course-ci.yml` installs pinned top-level CPU requirements on Python 3.12, checks dependency consistency, records the complete resolved environment, validates every notebook, runs the reference/unit suite, executes every worked notebook, and builds/runs a non-root inference container. Artifacts are retained for seven days, including failures. Jobs have time limits and cancel superseded runs. They use read-only repository permissions and do not need API keys or paid model services.

The top-level pins are not a cryptographically hashed transitive lock. Record the resolved environment attached to the exact run for research reproducibility. macOS, CUDA, external model downloads and true multi-host training require their own environment-specific validation.

## Graduation is a separate review

A course release candidate needs a passing clean-environment CI run and review of failure reports. A learner needs all six mastery gates, defended derivations and an independently explained capstone. A paper reproduction needs scientifically appropriate evidence and a stated scale gap, even when its code passes. No check in this repository grants an accredited credential or qualifies a synthetic detector for operational hardware.
