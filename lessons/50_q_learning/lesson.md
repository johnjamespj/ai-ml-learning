# Lesson 50: Value functions and Q-learning

## Value
V(s) estimates expected future return from state s.

## Action value
Q(s,a) estimates expected return after taking action a in state s.

## Bellman idea
Useful values are recursively related to immediate reward plus future value.

## Q-learning update
```text
Q(s,a) <- Q(s,a) + alpha * [
    r + gamma * max_a' Q(s',a') - Q(s,a)
]
```

The bracketed quantity is a temporal-difference error.

## Exploration
An epsilon-greedy policy:
- random action with probability epsilon
- otherwise choose best known action

## Exercise
Implement tabular Q-learning on FrozenLake or another small discrete environment.

Track:
- episode return
- success rate
- epsilon
- Q-table evolution

## Questions
Why is exploration necessary? What happens if gamma is near zero? Why can sparse rewards be difficult?
