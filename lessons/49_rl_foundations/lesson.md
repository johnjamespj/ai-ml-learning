# Lesson 49: Reinforcement learning foundations

## RL vocabulary
- agent
- environment
- observation/state
- action
- reward
- return
- policy
- episode
- discount factor

The agent repeatedly interacts:

state -> action -> environment -> reward + next state

## Markov decision process
An MDP formalizes:
- states S
- actions A
- transition dynamics
- reward function
- discount factor gamma

## Return
Discounted return from time t:

G_t = r_t + gamma r_(t+1) + gamma^2 r_(t+2) + ...

## Exercise
Use Gymnasium with a simple environment. Run random actions and log observation, reward, termination and episode return.

## Key lesson
RL learns from consequences, not labeled target outputs.
