# Lesson 70: Graph neural networks

Some data is naturally represented as nodes and edges:
- social networks
- molecules
- communication networks
- knowledge graphs
- sensor networks

## Message passing
A basic graph neural network repeatedly:
1. gathers information from neighboring nodes;
2. aggregates it;
3. updates each node representation.

Conceptually:

h_v' = update(h_v, aggregate({h_u : u in neighbors(v)}))

## Tasks
- node classification
- edge prediction
- graph classification

## Exercise
Implement one simple message-passing step with an adjacency matrix and NumPy before using a graph library.

## Important
Graph structure encodes assumptions about which relationships matter.
