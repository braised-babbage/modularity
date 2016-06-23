import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

def modmatrix(W,alpha=1.0):
    assert sp.sparse.isspmatrix(W), "Expected a sparse matrix."
    n,_ = W.shape
     
    e = np.ones(n)
    # The degree of vertex i is \sum_j W_{ij}, 
    # so the degree vector is simply W*e
    d = W*e
    # m is the total edge weight
    m = (0.5)*np.dot(e,d)
    
    # s is the sum of alpha-degrees
    d_alpha = d**(alpha)
    s = np.dot(e,d_alpha)

    P = np.outer(d_alpha,d_alpha.T) 
    Q = W / (2*m) - P / (s**2) # This is stored as a *dense* matrix.
    
    return Q

def modlinop(W,alpha=1.0):
    assert sp.sparse.isspmatrix(W), "Expected a sparse matrix."
    n,_ = W.shape
     
    e = np.ones(n)
    # The degree of vertex i is \sum_j W_{ij}, 
    # so the degree vector is simply W*e
    d = W*e
    # m is the total edge weight
    m = (0.5)*np.dot(e,d)
    
    # s is the sum of alpha-degrees
    d_alpha = d**(alpha)
    s = np.dot(e,d_alpha)

    
    def mv(v):
        return W*v / (2*m) - np.dot(d_alpha,v)*d_alpha / (s**2)
    def Mmv(v):
        return d*v
    
    return LinearOperator((n,n), matvec=mv), LinearOperator((n,n), matvec=Mmv)


def mod_labels(Q,c):
    # input should be a n-vector of integer labels in the range [0,n)
    m,n = Q.shape
    E = np.identity(n)
    C = E[c]
    M = np.inner(C,C)
    return np.sum(np.multiply(Q,M))
