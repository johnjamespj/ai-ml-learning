import importlib, os
import pytest
torch=pytest.importorskip("torch")
from torch import nn
m=importlib.import_module(f"labs.lab05_pytorch_training.{os.getenv('LAB_TARGET','solution')}")

def test_shapes():
    model=m.MLP()
    X,y=m.make_data(10,0)
    assert model(X).shape==(10,2)
    assert y.dtype==torch.long

def test_training_improves():
    torch.manual_seed(0)
    model=m.MLP()
    X,y=m.make_data(256,1)
    opt=torch.optim.Adam(model.parameters(),lr=.03)
    loss_fn=nn.CrossEntropyLoss()
    before=m.accuracy(model,X,y)
    for _ in range(150):
        m.train_step(model,X,y,opt,loss_fn)
    after=m.accuracy(model,X,y)
    assert after > before
    assert after > .85
