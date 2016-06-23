import cvxopt
import numpy as np
import scipy as sp
from modsdp.modularity import *
from sklearn.preprocessing import normalize


def sdp(W,k,solver = 'cvxopt', rounding = 'gw', round_iters = 1000):
    solvers = {'cvxopt' : sdpmod, 'rbr' : rbr, 'rand_rbr' : random_rbr}
    rounders = {'gw2' : goemans,
                'gw' : goemansK,
                'by' : bertyeK,
                'frieze' : friezeK}
    Q = modmatrix(W)
    X = solvers[solver](Q)
    c = rounders[rounding](Q,X,k=k, l = round_iters)
    return c

def sdpmod(Q):
    # The G matrix encodes the diagonal constraint on Z.
    def makeG(n):
        I = []
        J = []
        for i in range(0,n):
            I += [i]
            J += [n*i + i]
        return cvxopt.spmatrix(1.0, I, J, size=(n,n*n)).T

    m,n = Q.shape
    # Our constraint: Z_{ii} + c_i = 0
    c = cvxopt.matrix(-np.ones(n))
    # Sign convention.
    h = -cvxopt.matrix(Q)
    G = makeG(n)
    # Use the default solver for the primal-dual problem.
    sol = cvxopt.solvers.sdp(c, Gs=[G], hs = [h])
    # We only want the dual solution Z.
    return np.array(sol['zs'][0])

def rbr_update(M,X,i,nu=0.0):
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[i] = False

    
    B = X[mask,:][:,mask]

    c = 2*M[mask,i]
    Bc = B.dot(c)
    gamma = (c.T).dot(Bc)
    if gamma > 0:
        y = - np.sqrt((1-nu)/gamma)*Bc
    else:
        y = np.zeros(c.shape)
    X[mask,i] = y
    X[i,mask] = y

def rbr(M,eps_tol=1e-3, log = None):
    M = -np.array(M)
    n,m = M.shape
    assert(n==m)
    X = np.eye(n) # this initialization is IMPORTANT
    f = np.einsum('ij,ij', M, X)
    stopped = False
    k = 0
    while not stopped:
        k += 1
        for i in range(n):
            rbr_update(M,X,i,nu=0.1)
        f_old = f
        f = np.einsum('ij,ij', M, X)
        delta = (f_old - f) / max(f_old,1.0)
        if log:
            log.write("%d. delta = %f" % (k,delta))
        stopped = delta < eps_tol
    return X

def random_rbr(M, iters=50, log = None):
    M = -np.array(M)
    n,m = M.shape
    assert(n==m)
    X = np.eye(n) # this initialization is IMPORTANT
    f = np.einsum('ij,ij', M, X)
    for k in range(iters):
        k += 1
        for i in range(n):
            rbr_update(M,X,np.random.randint(n),nu=0.1)
        f_old = f
        f = np.einsum('ij,ij', M, X)
        delta = (f_old - f) / max(f_old,1.0)
        if log:
            log.write("%d. delta = %f" % (k,delta))
    return X


def bertyeK(Q,X,k=10,l=20):
    m,n = X.shape
    L = np.linalg.cholesky(X)
    
    E = np.identity(k)
    S = normalize(E - 1./k)
    projS = make_projS(k)

    maxq = 0.0
    for i in range(0,l):
        # rows of V are iid ~ N(0, identity(k))
        V = np.random.multivariate_normal(np.zeros(k), np.identity(k), n)
        V = np.dot(L,V)
        # project rows of V onto S
        c = projS(V)
        q = mod_labels(Q,c)
        if q > maxq:
            maxq = q
            maxc = c

    return maxc

def goemans(Q,Z,k = 2, l=1000):
    """ Apply the Goemans-Williamson rounding procedure.
        Q is the modularity matrix, Z is the relaxed solution.
        The rounding is randomized, iterate l times and return the best result. """
    m,n = Z.shape

    V = np.linalg.cholesky(Z)
    # goemans calls for rho to be a unit vector, but that isn't necessary
    rhos = np.random.multivariate_normal(np.zeros(m), np.identity(m),l) 
    zs = np.sign(np.dot(V,rhos.T)) # columns represent cluster membership
    mods = np.diag(np.dot(np.dot(zs.T,Q),zs))
    i = np.argmax(mods)
    c = ((zs[:,i]+1)/2).astype(int)
    return c

def goemansK(Q,X,k=10,l=1000):
    m,n = X.shape

    V = np.linalg.cholesky(X).T
    
    
    projS = make_projS(k)
        

    maxq = 0.0
    for i in range(0,l):
        Q1,_ = sp.linalg.qr(np.random.randn(m,k), mode='economic')
        V1 = np.dot(Q1.T, V)
        
        c = projS(V1.T)
        # modularity
        q = mod_labels(Q,c)
        if (q > maxq):
            maxq = q
            maxc = c
    return maxc

def make_projS(k):
    E = np.identity(k)
    S = normalize(E - 1./k,axis=0)
    def projS(V):
        # project rows of V onto columns of V
        D = np.dot(V,S)
        return np.argmax(D, axis=1)
    return projS


def friezeK(Q,X,k=10,l=1000):
    m,n = X.shape

    V = np.linalg.cholesky(X).T

    maxq = 0.0
    for i in range(0,l):
        # k random points in
        S = normalize(np.random.rand(m,k),axis=0).T
        
        
        c = np.argmax(S.dot(V),axis=0)
        # modularity
        q = mod_labels(Q,c)
        if (q > maxq):
            maxq = q
            maxc = c
    return maxc
