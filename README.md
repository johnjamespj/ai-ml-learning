# AI/ML Learning Lab

A notebook-first course combining derivations, runnable experiments, implementation labs, landmark papers and an end-to-end engineering capstone.

## Start with a path, not an encyclopedia

Follow the **[42-lesson Core Track](COURSE_TRACKS.md)**, then take the seven-lesson signal/radar specialization or select from 29 advanced electives. The full collection retains 78 lessons, 17 math companions and 21 paper notebooks.

```bash
git clone https://github.com/johnjamespj/ai-ml-learning.git
cd ai-ml-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu.txt
python -m ipykernel install --user --name ai-ml-course
jupyter lab
```

These are Linux CPU instructions. For macOS or CUDA use the matching official PyTorch wheels rather than the Linux CPU requirements file. Select the `ai-ml-course` kernel in Jupyter.

## Learn, implement, then assess

Begin at [Lesson 00](lessons/00_setup/lesson.ipynb), use the linked [mathematical framework](math/README.md), and work through the [study plan](STUDY_PLAN.md). Make an independent workspace with `python tools/make_student_workspace.py --output work`.

The course includes **12 implementation labs, six cumulative exams, and eight mathematical problem sets**. Starter work is separate from instructor answers. Passing the reference suite checks the course, not your understanding. Exams remain NOT GRADED until you attempt them; [the rubric](assessment/RUBRIC.md) requires derivation, coding, debugging, experimental design and defense.

## Research and engineering

The [paper track](papers/NOTEBOOK_TRACK.md) now exports source/configuration provenance, figures, diagnostics and clearly marked ablation/interpretation status. Use [paired-seed evidence recording](papers/EXPERIMENTS.md) rather than choosing a lucky curve.

The **[flagship RF detection capstone](capstones/rf_detection/capstone.ipynb)** compares energy and matched-filter baselines, feature ML, a spectrogram CNN and a temporal Transformer. It separates fitting, validation, probability calibration, threshold setting and testing; reports Pd/Pfa against SNR and shifts; exports a safe model artifact; serves it through FastAPI; and includes Docker parity and monitoring checks.

## Trust requires execution evidence

```bash
python tools/validate_course.py
python -m pytest -q
python tools/execute_notebooks.py --group smoke
python tools/execute_notebooks.py --group all --workers 2
```

GitHub Actions installs a pinned CPU environment, validates structure and links, runs tests, executes every worked notebook in a fresh kernel, and tests the real inference container. Errors fail the workflow and reports are retained. GPU/distributed/download paths are explicitly separate, not quietly counted as tested.

Read the [quality contract](docs/QUALITY.md) and [implementation references](docs/SOURCES.md). Execution, scientific reproduction, and learner mastery are different standards. Synthetic RF results are not hardware qualification.
