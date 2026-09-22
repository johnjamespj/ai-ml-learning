import numpy as np

def assign_clusters(X,centroids):
    d=((X[:,None,:]-centroids[None,:,:])**2).sum(axis=2)
    return d.argmin(axis=1)

def update_centroids(X,labels,k):
    return np.vstack([X[labels==i].mean(axis=0) for i in range(k)])

def kmeans(X,k,steps=100,seed=0):
    rng=np.random.default_rng(seed)
    centroids=X[rng.choice(len(X),size=k,replace=False)].astype(float).copy()
    for _ in range(steps):
        labels=assign_clusters(X,centroids)
        if any(np.sum(labels==i)==0 for i in range(k)):
            break
        new=update_centroids(X,labels,k)
        if np.allclose(new,centroids):
            centroids=new; break
        centroids=new
    return labels,centroids

def pca_fit_transform(X,n_components):
    mean=X.mean(axis=0)
    Xc=X-mean
    cov=np.cov(Xc,rowvar=False)
    vals,vecs=np.linalg.eigh(cov)
    order=np.argsort(vals)[::-1]
    vals=vals[order]
    components=vecs[:,order[:n_components]].T
    Z=Xc@components.T
    return Z,components,mean,vals[:n_components]

def pca_inverse(Z,components,mean):
    return Z@components+mean
