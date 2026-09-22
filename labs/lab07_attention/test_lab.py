import importlib, os, pytest
torch=pytest.importorskip("torch")
m=importlib.import_module(f"labs.lab07_attention.{os.getenv('LAB_TARGET','solution')}")

def test_attention_shapes_and_rows():
    Q=torch.randn(2,3,5,4); K=torch.randn(2,3,5,4); V=torch.randn(2,3,5,7)
    out,w=m.scaled_dot_product_attention(Q,K,V)
    assert out.shape==(2,3,5,7)
    assert w.shape==(2,3,5,5)
    assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-5)

def test_causal_mask():
    T=4
    Q=K=torch.eye(T).reshape(1,1,T,T)
    V=torch.arange(T,dtype=torch.float32).reshape(1,1,T,1)
    mask=torch.tril(torch.ones(T,T,dtype=torch.bool)).reshape(1,1,T,T)
    _,w=m.scaled_dot_product_attention(Q,K,V,mask)
    assert torch.all(w[0,0].triu(1)==0)
