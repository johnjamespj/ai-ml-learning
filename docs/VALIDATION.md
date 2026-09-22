# Validation record: 22 September 2026

Source baseline: `f005e358950bee1ee868664babf30f7c907257d7`.

## Actual local execution

The first audit of the 118 pre-existing worked notebooks recorded 98 passes,
18 failures, and two safety holds (automatic environment/server launch and model downloads).
The intermediate repaired run included one kernel-start infrastructure failure; it was retained,
then the complete suite was repeated with two concurrent kernels.

The final run executed **119 worked notebooks**, including the RF capstone, in independent
kernels and temporary working directories: **112 passed; 7 passed their CPU path with
optional hardware/download paths explicitly not run; 0 failed**. Source hashes were unchanged
during the run. Python was 3.13.5, not the separate 3.12 GitHub Actions environment.

**60 automated tests passed**. These test reference implementations and engineering contracts,
not the learner's completion of the course. All six exams and eight problem sets remain
ungraded until the learner implements their answers and completes human-reviewed components.

## Measured capstone evidence

Five models were run across three seeds, four conditions and six SNR values: **360 measured
rows**, with train, validation, probability calibration, threshold calibration and final test
samples separated. The neural models did not automatically beat the matched-filter baseline.
See `LOCAL_VALIDATION.json` for scope, versions, hashes and the measured local NumPy latency.
The full experiment can be regenerated using the command in the capstone README.

## Boundaries

CPU execution is not CUDA, actual distributed-training, downloaded pretrained-model, or
Docker-runtime validation. The GitHub Actions workflow separately installs a fresh pinned
Python 3.12 environment and actually builds/runs Docker to test HTTP parity.
Its status must be read from the live run, not inferred from this local report.

Passing notebook execution does not establish reproduction of an original paper's benchmark,
statistical superiority, actual learner mastery, or real-world radar qualification.
