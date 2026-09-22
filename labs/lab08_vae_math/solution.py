import torch

def reparameterize(mu,logvar,eps=None):
    std=torch.exp(0.5*logvar)
    if eps is None:
        eps=torch.randn_like(std)
    return mu+std*eps

def kl_standard_normal(mu,logvar):
    kl=-0.5*(1+logvar-mu.pow(2)-logvar.exp())
    return kl.sum(dim=-1).mean()
