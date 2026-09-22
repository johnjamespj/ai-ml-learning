# Lesson 29: Validation, checkpoints and early stopping

A serious training run needs more than a loop that prints loss.

## Validation pass
```python
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)

val_acc = correct / total
```

## Checkpoint
```python
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "val_loss": val_loss,
}, "checkpoint.pt")
```

## Restore
```python
ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
optimizer.load_state_dict(ckpt["optimizer_state"])
```

## Track
At minimum record:
- train loss
- validation loss
- chosen metric
- learning rate
- epoch
- seed
- data split
- model configuration

## Exercise
Add early stopping and save only the best validation checkpoint. Then intentionally overfit a tiny dataset and verify your validation curve catches it.
