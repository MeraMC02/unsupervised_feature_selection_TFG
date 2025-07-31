import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, NearestNeighbors

def knn(X,k):
    Nk = kneighbors_graph(X,n_neighbors=k,mode="connectivity",include_self=False)
    return Nk.maximum(Nk.T)

def symmetrize(S, mode="average"):
    if mode == "average":
        return 0.5 * (S + S.T)
    elif mode == "max":
        return S.maximum(S.T)
    else:
        raise ValueError("mode must be 'average' or 'max'")

def affinityGraph(Nk,X,sigma=None,sym_mode="average"):
    coo = Nk.tocoo()
    difference = X[coo.row] - X[coo.col]
    distance = (difference**2).sum(axis=1)

    if sigma is None:                          
        sigma2 = np.median(distance) + 1e-12
    else:
        sigma2 = float(sigma)**2 + 1e-12

    weights = np.exp(-distance/(2*sigma**2))
    S = sparse.csr_matrix((weights, (coo.row, coo.col)), shape=Nk.shape)
    return symmetrize(S,mode=sym_mode)

def adaptiveAffinityGraph(Z, k, lambdaVal, eps):
    n = Z.shape[0]
    knn =  NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    knn.fit(Z)
    dist, idx = knn.kneighbors(Z, return_distance=True)

    dk1  = dist[:, [k]]
    dsum = dist[:, :k].sum(axis=1, keepdims=True)
    denom = k*dk1 - dsum
    denom[denom < eps] = eps

    s = (lambdaVal*(dk1-dist[:, :k]))/denom
    np.maximum(s, 0.0, out=s)

    rows = np.repeat(np.arange(n), k)
    cols = idx[:, :k].ravel()
    data = s.ravel()

    S = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    S = symmetrize(S)

    gamma = float(np.mean((k*dk1[:,0] - dsum[:,0])/(2.0*lambda_val)))
    return S, gamma


def laplacian(S):
    d = np.asarray(S.sum(axis=1)).ravel()
    D = sparse.diags(d)
    return D - S
