# Student entry point

Start with the [Core Track](../COURSE_TRACKS.md), not a list of 78 equally urgent topics.

```bash
python tools/make_student_workspace.py --output work
```

The workspace contains 12 lab starters, their locally targeted tests, and six exam notebooks. No reference solution files are copied. It refuses to overwrite an existing workspace. Implement `work/labs/<lab>/solution.py`, then run its tests. The filename is the learner's submission, not an answer key.

```bash
python -m pytest --import-mode=importlib work/labs/lab01_linear_regression/test_lab.py -q
```

The canonical lesson and paper notebooks are worked demonstrations. They can run before you have learned to implement the method; their execution is not a mastery score. The two notebooks under `notebooks/` explicitly use reference implementations for demonstration.

For an exam, implement the notebook functions or its adjacent `submission.py`. Then use:

```bash
python tools/grade_submission.py --exam 01_foundations --submission work/exams/01_foundations/submission.py --output work/exams/01_foundations/grade.json
```

The default exam and problem-set state is **NOT GRADED**. Passing numerical checks still requires a human review of derivations, debugging, design and oral defense. See the [rubric](../assessment/RUBRIC.md).
