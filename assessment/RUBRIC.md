# Mastery rubric

Each of the six exams has 100 points: derivation 20, coding 30, debugging 15, experimental design 20, and defense 15.

A pass requires at least 80 total points, at least half the points in every component, passing numerical checks, and no critical errors. Critical errors include train/test leakage, unsupported claims of reproduction, fabricated evidence, an unrecognized shape mistake invalidating the result, and serving a different preprocessing pipeline from the one evaluated.

Award derivation points for explicitly defined variables/shapes (5), justified steps (10), and assumptions or boundary conditions (5). Coding points require correct numerical behavior (15), edge-case/numerical stability handling (10), and clear reproducible interfaces (5). Debugging requires locating the cause, explaining its effect, and producing an independent regression test (5 each). Design requires valid splits (5), suitable baselines and metrics (5), controlled comparisons and uncertainty (5), and failure/robustness analysis (5). Defense requires an unaided explanation, an alternative, and a stated limitation (5 each).

Numerical checks cannot judge a proof, novelty claim, or experimental design. The grader deliberately returns `not_yet_assessed` when only coding checks pass.

A human review JSON can be supplied with `--review`:

```json
{
  "reviewer": "name of actual reviewer",
  "date": "YYYY-MM-DD",
  "points": {"derive": 0, "code": 0, "debug": 0, "design": 0, "defend": 0},
  "no_critical_errors": false,
  "notes": "Replace with evidence-backed evaluation. Do not prefill a pass."
}
```

After a revision, retain the first attempt and its errors. Reassess with changed data/seeds or a parallel problem rather than repeatedly rehearsing the same visible fixture. This repository provides learning assessments, not an accredited qualification.
