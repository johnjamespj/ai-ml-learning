# Lesson 69: Recommender systems

Recommendation is prediction over user-item interactions.

## Main approaches
- popularity baseline
- content-based filtering
- collaborative filtering
- matrix factorization
- neural recommenders

## Matrix factorization
Represent user u and item i with latent vectors:

score(u,i) = p_u dot q_i

Learn the vectors from observed interactions.

## Implicit vs explicit feedback
Ratings are explicit. Clicks, views and purchases are implicit and often ambiguous.

## Evaluation
Random splits can be misleading. Prefer time-aware evaluation when simulating future recommendations.

Metrics may include:
- Recall@k
- Precision@k
- MAP
- nDCG

## Exercise
Build a tiny matrix-factorization recommender using NumPy or PyTorch and compare it with a popularity baseline.
