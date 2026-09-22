import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,2))

    def forward(self,x):
        return self.net(x)

def make_data(n=256,seed=0):
    g=torch.Generator().manual_seed(seed)
    X=torch.randn(n,2,generator=g)
    y=((X[:,0]*X[:,1])>0).long()
    return X,y

def train_step(model,X,y,optimizer,loss_fn):
    model.train()
    optimizer.zero_grad()
    logits=model(X)
    loss=loss_fn(logits,y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

def accuracy(model,X,y):
    model.eval()
    with torch.no_grad():
        pred=model(X).argmax(1)
        return float((pred==y).float().mean())
