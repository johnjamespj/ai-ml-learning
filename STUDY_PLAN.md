# Study plan

## The normal cycle

Read the lesson's mathematical framework, derive its central equation, predict shapes and limiting behavior, run the worked notebook, change one assumption, measure the result, and explain it. Complete the corresponding independent lab before reading its reference implementation.

Use [Core/Advanced/Radar tracks](COURSE_TRACKS.md) to avoid treating all 78 topics as simultaneous prerequisites. Revisit math companions when needed rather than rereading the entire math sequence before every lesson.

## Six cumulative gates

1. Foundations: scientific Python, gradients, probability and estimation. Pass Exam 01 and the early math problem sets.
2. Classical ML: build a leakage-safe benchmark against a dummy baseline. Pass Exam 02.
3. Neural networks: implement the NumPy MLP, gradient-check it and explain XOR. Pass Exam 03 and the backprop paper defense.
4. PyTorch and vision: train/evaluate/checkpoint a model and diagnose failures. Pass Exam 04.
5. Transformers and LLM systems: implement attention and masks, reproduce an order-sensitive experiment and evaluate retrieval separately. Pass Exam 05.
6. Systems: complete the synthetic RF capstone or an equivalently defended end-to-end system. Pass Exam 06 and the math synthesis workshop.

Every exam mixes derivation, implementation, debugging, experimental design and defense. Use the [rubric](assessment/RUBRIC.md); numerical success does not complete a written or oral assessment.

## Paper work

Use the [21 paper notebooks](papers/NOTEBOOK_TRACK.md), state the scale gap, predeclare comparisons, run paired seeds, and retain failures. Automatic evidence exports distinguish measured values from unperformed ablations. A smaller experiment can fail to reproduce a trend; report that result honestly.

## Final evidence packet

Keep your derivations, six reviewed exam records, starter-test results, paper ablation reports, model/data/configuration provenance, final test report, inference parity check and deployment/monitoring limitations. A polished demo alone is insufficient.
