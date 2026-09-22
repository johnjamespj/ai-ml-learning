def train_step(model, X, y, optimizer):
    raise NotImplementedError("One complete optimization step")

def evaluate_accuracy(model, X, y):
    raise NotImplementedError("Evaluate and restore the previous mode")
