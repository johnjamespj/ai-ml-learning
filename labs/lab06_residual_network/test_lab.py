import importlib, os, pytest
torch=pytest.importorskip("torch")
m=importlib.import_module(f"labs.lab06_residual_network.{os.getenv('LAB_TARGET','solution')}")

def test_shape_and_gradient():
    block=m.ResidualBlock(8)
    x=torch.randn(4,8,requires_grad=True)
    y=block(x)
    assert y.shape==x.shape
    y.sum().backward()
    assert x.grad is not None
