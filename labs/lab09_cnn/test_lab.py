import importlib, os, pytest
torch=pytest.importorskip("torch")
m=importlib.import_module(f"labs.lab09_cnn.{os.getenv('LAB_TARGET','solution')}")

def test_output_shape():
    model=m.SmallCNN(10)
    x=torch.randn(4,1,28,28)
    assert model(x).shape==(4,10)

def test_parameter_count():
    model=m.SmallCNN()
    assert m.count_parameters(model)>0
