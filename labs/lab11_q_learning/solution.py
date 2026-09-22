import numpy as np

class ChainEnv:
    def __init__(self,n_states=5,max_steps=20):
        self.n_states=n_states; self.max_steps=max_steps
    def reset(self):
        self.s=0; self.steps=0; return self.s
    def step(self,action):
        self.steps+=1
        self.s=max(0,min(self.n_states-1,self.s+(1 if action else -1)))
        done=self.s==self.n_states-1 or self.steps>=self.max_steps
        reward=1.0 if self.s==self.n_states-1 else 0.0
        return self.s,reward,done

def train_q(episodes=1000,alpha=.2,gamma=.95,epsilon=.2,seed=0):
    rng=np.random.default_rng(seed)
    env=ChainEnv()
    Q=np.zeros((env.n_states,2))
    for _ in range(episodes):
        s=env.reset()
        while True:
            a=int(rng.integers(2)) if rng.random()<epsilon else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            ns,r,done=env.step(a)
            terminated = ns == env.n_states-1
            target=r if terminated else r+gamma*np.max(Q[ns])
            Q[s,a]+=alpha*(target-Q[s,a])
            s=ns
            if done: break
    return Q
