# Lab 11: Q-Learning Without Gym

Linked lessons: 49-51. Paper companion: DQN.

Use a five-state chain so the Bellman update is visible rather than hidden behind a game.

## Task
Implement epsilon-greedy action selection and tabular Q-learning.

## Tests
```bash
LAB_TARGET=starter pytest labs/lab11_q_learning/test_lab.py -q
```

## Experiments
Sweep alpha, gamma and epsilon-decay behavior. Plot Q-values by state.

## Explain
Point to the exact line where immediate reward and estimated future reward are combined.
