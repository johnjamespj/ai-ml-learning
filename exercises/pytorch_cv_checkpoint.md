# PyTorch and Computer Vision Checkpoint

1. What is the key difference between a NumPy array and a PyTorch tensor with `requires_grad=True`?
2. What does `.backward()` compute?
3. Why must gradients usually be cleared before another optimizer step?
4. What role does `nn.Module` play?
5. Why use Dataset and DataLoader?
6. How does batch size affect memory and optimization?
7. Why must model and data be on the same device?
8. Why does GPU transfer overhead matter?
9. What does `model.eval()` change?
10. Why use `torch.no_grad()` for evaluation?
11. What belongs in a training checkpoint?
12. Why intentionally overfit a tiny batch while debugging?
13. What is a convolution kernel?
14. Why does parameter sharing matter?
15. What is a receptive field?
16. What do stride and padding change?
17. Why can pooling help?
18. What is data augmentation?
19. When can augmentation corrupt labels?
20. What is transfer learning?
21. Why can a spectrogram be treated as an image-like tensor?
22. Which STFT parameters affect time/frequency resolution?
23. Why can naive random splitting make synthetic signal results look too good?
24. How would you test robustness to SNR shift?
25. Explain the path from waveform -> spectrogram -> CNN -> logits -> loss -> gradient update.
