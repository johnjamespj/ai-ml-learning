# Lesson 31: Convolutions and CNN intuition

## Why convolutions
Images and many signal representations have local structure. Convolutions apply a small learned filter across space.

## 2-D convolution intuition
A kernel slides across an image and produces responses based on local patterns.

```python
import torch
from torch import nn

conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1,
)

x = torch.randn(16, 1, 28, 28)
y = conv(x)
print(y.shape)
```

## Terms
- channels
- kernels/filters
- feature maps
- stride
- padding
- receptive field
- pooling

## Why parameter sharing matters
The same filter is reused across locations. This greatly reduces parameter count compared with a fully connected layer over all pixels.

## Signal connection
A spectrogram is a 2-D time-frequency representation, so CNNs can learn local patterns such as bursts, chirps and narrowband interference.

## Exercise
Manually compute one small convolution using a 5x5 input and 3x3 kernel, then compare with PyTorch.
