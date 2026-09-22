import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Retriever:
    def __init__(self,documents):
        self.documents=list(documents)
        self.vectorizer=TfidfVectorizer()
        self.matrix=self.vectorizer.fit_transform(self.documents)
    def search(self,query,k=3):
        q=self.vectorizer.transform([query])
        scores=(self.matrix@q.T).toarray().ravel()
        return np.argsort(scores)[::-1][:k].tolist()

def recall_at_k(rankings,relevant,k):
    vals=[]
    for ranking,rel in zip(rankings,relevant):
        rel=set(rel)
        vals.append(len(set(ranking[:k])&rel)/len(rel) if rel else 1.0)
    return float(np.mean(vals))

def mrr(rankings,relevant):
    vals=[]
    for ranking,rel in zip(rankings,relevant):
        rel=set(rel); rr=0.0
        for rank,idx in enumerate(ranking,1):
            if idx in rel:
                rr=1/rank; break
        vals.append(rr)
    return float(np.mean(vals))
