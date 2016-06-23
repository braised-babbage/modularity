import numpy as np
from .modularity import *
from scipy.sparse.linalg import eigsh

def spectral(W,k,round_iters=10):
    w,v = modspectralK(W,k)
    vs = vertex_vectors(w,v)
    Q = modmatrix(W)
    maxq = -1.1
    for i in range(round_iters):
        c = vector_partition(vs,k)
        q = mod_labels(Q,c)
        if q > maxq:
            maxc = c
            maxq = q
    return maxc

def modspectralK(W,k=2):
    Q,M = modlinop(W)

    w,v = eigsh(Q,M=M,k=k)
    
    return w,v

def vertex_vectors(w,v):
    w = np.maximum(w,0.0)
    return np.dot(v,np.diag(np.sqrt(w)))

def vector_partition(vs,k):
    # vs is a n x p matrix of n p-dimensional vertex vectors
    n,p = vs.shape
    groups = vs[np.random.choice(n,k)] 
    groups[-1,:] -= np.sum(groups,axis=0) # group vectors sum to 0
    comms = np.zeros(400,dtype='int64')
    while True:
        ips = np.dot(groups, vs.T)
        old = comms
        comms = np.argmax(ips,axis=0)
        if np.array_equal(old,comms):
            break
        groups[:] = 0.0
        for i in range(0,n):
            groups[comms[i],:] += vs[i,:]
    return comms #dict(enumerate(comms))
