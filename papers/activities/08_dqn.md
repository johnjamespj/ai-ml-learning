# Paper Activity 08: Deep Q-Networks

**Paper:** Mnih et al., "Human-level control through deep reinforcement learning" (Nature, 2015).

## Educational reproduction
Use a small Gymnasium task rather than Atari.

Build DQN with:
- replay buffer
- target network
- epsilon-greedy exploration
- minibatch TD updates

## Ablation
Remove the target network, then remove replay. Compare stability across multiple seeds.

## Explain
What instability does each engineering choice address?
