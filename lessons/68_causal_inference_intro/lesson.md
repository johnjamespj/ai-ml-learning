# Lesson 68: Causal inference introduction

Prediction asks:
"What is likely to happen?"

Causal inference asks:
"What would happen if we changed something?"

These are different questions.

## Core ideas
- correlation vs causation
- confounders
- treatment
- outcome
- counterfactual
- intervention
- randomized experiments
- observational data limitations

## DAG intuition
Directed acyclic graphs provide a language for assumed causal structure.

## Example
Ice-cream sales may correlate with drowning incidents because temperature influences both. A predictive association does not imply that reducing ice-cream sales reduces drownings.

## Exercise
Create three synthetic variables with a confounder. Show a misleading raw correlation and then control for the confounder.

## Rule
Do not interpret feature importance or predictive coefficients as causal effects without a valid causal design.
