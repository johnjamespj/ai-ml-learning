import importlib, os, numpy as np
m=importlib.import_module(f"labs.lab11_q_learning.{os.getenv('LAB_TARGET','solution')}")

def test_chain():
    e=m.ChainEnv()
    s=e.reset()
    for _ in range(4):
        s,r,d=e.step(1)
    assert d and r==1 and s==4

def test_q_prefers_right():
    Q=m.train_q(episodes=2000,epsilon=.3,seed=1)
    assert np.all(Q[:-1,1] >= Q[:-1,0])
    assert Q[0,1]>0
