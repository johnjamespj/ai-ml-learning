# Lesson 16: Clustering

## Goal
Learn how algorithms can discover structure without labels.

## k-means
k-means alternates between:
1. assigning each sample to its nearest centroid;
2. replacing each centroid with the mean of its assigned samples.

It minimizes within-cluster squared distance.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.2, random_state=42)
model = KMeans(n_clusters=4, n_init="auto", random_state=42)
labels = model.fit_predict(X)
print(model.cluster_centers_)
```

## From-scratch challenge
Implement centroid initialization, assignment, centroid update, convergence detection and inertia using NumPy.

## Limits
Clusters are not automatically real-world categories. k-means prefers roughly compact clusters and requires choosing k.

## Beyond k-means
Study hierarchical clustering and DBSCAN. DBSCAN can discover irregular cluster shapes and mark sparse observations as noise.

## Experiments
Create blob, moon and varying-density datasets. Compare k-means and DBSCAN. Scale features first and observe what changes.

## Questions
What is unsupervised learning? Why is cluster evaluation difficult without labels? Why can scaling change the discovered structure?
