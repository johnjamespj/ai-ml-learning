def relu_backward(upstream, z):
    raise NotImplementedError("Apply the local Jacobian")

def cross_entropy_logits(logits, y):
    raise NotImplementedError("Use stable log-sum-exp")
