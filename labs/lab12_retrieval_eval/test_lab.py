import importlib, os
m=importlib.import_module(f"labs.lab12_retrieval_eval.{os.getenv('LAB_TARGET','solution')}")

def test_retrieval():
    docs=["gradient descent optimizes a loss","attention uses query key value","k means clusters samples"]
    r=m.Retriever(docs)
    assert r.search("query key attention",1)[0]==1

def test_metrics():
    rankings=[[2,1,0],[0,2,1]]
    relevant=[{1},{0}]
    assert m.recall_at_k(rankings,relevant,1)==.5
    assert abs(m.mrr(rankings,relevant)-.75)<1e-9
