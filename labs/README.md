# Runnable Labs

The lessons explain ideas. The labs make you implement, test, break, measure, and defend them.

## Lab workflow
1. Read the linked lesson.
2. Open `starter.py`.
3. Replace each TODO.
4. Run the starter-targeted tests:
   ```bash
   LAB_TARGET=starter pytest labs/labXX_name/test_lab.py -q
   ```
5. Run the experiment in the lab README.
6. Only then inspect `solution.py`.
7. Write the requested explanation in your own notes.

By default, tests target the reference solution so the repository itself stays healthy:
```bash
pytest labs -q
```

## Labs
- Lab 01: linear regression and gradient descent
- Lab 02: logistic regression and thresholding
- Lab 03: k-means and PCA
- Lab 04: NumPy MLP and backpropagation
- Lab 05: PyTorch autograd/training loop

More labs should follow the same structure: starter code, reference solution, tests, experiment, and explanation.
