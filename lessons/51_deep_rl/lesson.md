# Lesson 51: Deep reinforcement learning

Tabular Q-learning breaks down when state spaces become large or continuous.

## DQN idea
Use a neural network Q_theta(s,a) to approximate action values.

Important stabilizers:
- replay buffer
- target network
- minibatch updates
- exploration schedule

## Policy methods
Instead of learning action values only, directly parameterize a policy pi_theta(a|s).

## Actor-critic
Use:
- actor: chooses actions
- critic: estimates value/action value

## Exercise
Train a small DQN on CartPole or another manageable task. Plot return and episode length.

## Caution
RL metrics can be noisy. Use multiple seeds and report distributions, not one lucky run.

## Drone connection
RL can be useful for control and navigation, but simulation validity, reward design, safety constraints and sim-to-real shift matter enormously.
