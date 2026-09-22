# Lesson 24: Making neural networks generalize

Training loss is not the destination.

## Techniques
Study train/validation curves, weight decay, dropout, early stopping, data augmentation, normalization and capacity control.

## Dropout intuition
During training, randomly suppress some activations. The network cannot rely on every feature pathway being present every time.

## Experiment
Train an intentionally oversized network on a small dataset. Record train and validation loss. Then separately add:
- weight decay;
- dropout;
- early stopping;
- more training data.

Compare the curves.

## Diagnosis
If both training and validation performance are poor, adding regularization is unlikely to solve the main problem. If training is excellent while validation degrades, investigate overfitting.

## Question
Why is "make the model bigger" not a universal solution?
