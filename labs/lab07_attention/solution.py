import math
import torch

def scaled_dot_product_attention(Q,K,V,mask=None):
    scores=Q@K.transpose(-2,-1)/math.sqrt(Q.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool or not mask.any(dim=-1).all():
            raise ValueError("Mask must be boolean and allow at least one key per row")
        scores=scores.masked_fill(~mask, float("-inf"))
    weights=torch.softmax(scores,dim=-1)
    return weights@V,weights
