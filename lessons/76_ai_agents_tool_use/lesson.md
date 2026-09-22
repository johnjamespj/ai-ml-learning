# Lesson 76: AI agents and tool use

An agentic system combines a model with a loop that can select actions, use tools and observe results.

## Components
- model
- instructions/policy
- tool definitions
- state/memory
- planning or action selection
- observations
- stopping condition
- evaluation

## Tool call pattern
```text
user goal
  -> model chooses tool/action
  -> environment executes
  -> observation returns
  -> model chooses next action
  -> final result
```

## Risks
- wrong tool selection
- cascading errors
- untrusted tool output
- excessive permissions
- loops
- poor stopping rules

## Exercise
Build a tiny deterministic tool-using agent with calculator/search-like local functions. Log each action and evaluate task completion separately from response style.

## Lesson
An agent is a system around a model, not a new kind of neural-network layer.
