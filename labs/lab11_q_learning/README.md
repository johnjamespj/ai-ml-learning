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

## Time-limit semantics

The 20-step limit truncates an episode but is not a terminal MDP state. The update zeros its bootstrap only on the goal state. A finite-horizon formulation would instead include remaining time in the state. Ties among greedy actions are randomized.
