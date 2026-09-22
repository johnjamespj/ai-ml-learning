import math
import torch

def scaled_dot_product_attention(Q,K,V,mask=None):
    scores=Q@K.transpose(-2,-1)/math.sqrt(Q.shape[-1])
    if mask is not None:
        scores=scores.masked_fill(~mask, float("-inf"))
    weights=torch.softmax(scores,dim=-1)
    return weights@V,weights
