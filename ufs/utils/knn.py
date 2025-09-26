import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

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

    weights = np.exp(-distance/(2*sigma2))
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

    gamma = float(np.mean((k*dk1[:,0] - dsum[:,0])/(2.0*lambdaVal)))
    return S, gamma


def laplacian(S,normalized=False, return_degree=False):
    d = np.asarray(S.sum(axis=1)).ravel()

    if not normalized:
        D = sparse.diags(d)
        L = D - S
        return(L,D) if return_degree else L

    d_mhalf = np.zeros_like(d, dtype=float)
    mask = d > 0
    d_mhalf[mask] = 1.0 / np.sqrt(d[mask])
    D_mhalf = sparse.diags(d_mhalf)
    L = sparse.identity(S.shape[0], format='csr') - (D_mhalf @ S @ D_mhalf)
    return L

def pairwise_sq_distances(A):
    sq = np.sum(A*A, axis=1, keepdims=True) # Calculamos la norma euclídea de las filas
    D = sq + sq.T - 2.0 * (A@A.T) # ||a_i - a_j||^2 = ||a_i||^2 + ||a_j||^2 - 2 a_i a_j
    np.maximum(D, 0.0, out=D) # Corregimos posibles fallos de redondeos negativos
    return D

def softmax(M, noNull):
    M -= M.max(axis=1, keepdims=True) #Aseguramos que no haya números grandes para que no se de un desborde
    np.exp(M, out=M) #Hacemos la exponencial 
    M /= (M.sum(axis=1, keepdims=True) + noNull) #Dividimos entre la suma de las exponenciales
    return M

def randomOrth(n,c,rng):
    Aux = rng.standard_normal((n, c))
    Y_p, _ = np.linalg.qr(Aux)            # columnas ortonormales
    Y_p = Y_p[:, :c]

    return Y_p