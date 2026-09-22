import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Retriever:
    def __init__(self,documents):
        # TODO
        raise NotImplementedError

    def search(self,query,k=3):
        # Return document indices ranked best first
        raise NotImplementedError

def recall_at_k(rankings,relevant,k):
    # relevant: list of sets of relevant doc indices
    raise NotImplementedError

def mrr(rankings,relevant):
    # TODO
    raise NotImplementedError
