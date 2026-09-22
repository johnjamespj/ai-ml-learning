# Lesson 36: RNNs, LSTMs and GRUs

## Recurrent idea
An RNN updates a hidden state using the current input and previous hidden state.

Conceptually:

h_t = tanh(W_x x_t + W_h h_(t-1) + b)

The same parameters are reused across time.

## PyTorch
```python
import torch
from torch import nn

rnn = nn.RNN(
    input_size=8,
    hidden_size=32,
    batch_first=True,
)

x = torch.randn(16, 50, 8)
output, h_last = rnn(x)

print(output.shape)
print(h_last.shape)
```

## Why vanilla RNNs struggle
Repeated multiplication through time can produce vanishing or exploding gradients.

LSTMs and GRUs add gating mechanisms that make long-range information easier to preserve.

## Exercise
Train an RNN, GRU and LSTM on the same synthetic sequence task. Compare parameter count, training speed and long-range performance.

## Signal connection
Use recurrent models for pulse sequences, temporal sensor behavior and learned representations of feature sequences.
