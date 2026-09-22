import importlib, os, pytest
torch=pytest.importorskip("torch")
m=importlib.import_module(f"labs.lab08_vae_math.{os.getenv('LAB_TARGET','solution')}")

def test_reparameterize_deterministic_eps():
    mu=torch.tensor([[1.,2.]])
    logvar=torch.zeros_like(mu)
    eps=torch.tensor([[.5,-.5]])
    z=m.reparameterize(mu,logvar,eps)
    assert torch.allclose(z,torch.tensor([[1.5,1.5]]))

def test_zero_kl_for_standard_normal():
    mu=torch.zeros(5,3); logvar=torch.zeros(5,3)
    assert torch.allclose(m.kl_standard_normal(mu,logvar),torch.tensor(0.))
