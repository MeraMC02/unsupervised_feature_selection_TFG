import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sklearn.utils import check_random_state
from .base import BaseEstimatorUFS 
from .utils.knn import affinityGraph, knn, laplacian, symmetrize, adaptiveAffinityGraph

class egcfs(BaseEstimatorUFS):

    def __init__(self, nFeaturesOut=None, alpha=1.0, lambdaArg=1.0, nClusters=2, k=5, maxIter=30,tol=1e-4,random_state=None):
        super().__init__(nFeaturesOut)
        self.alpha = alpha
        self.lambdaArg = lambdaArg
        self.nClusters=nClusters
        self.k = k
        self.maxIter = maxIter
        self.tol = tol
        self.random_state = random_state

    def _core(self,X):
        nSamples, nFeatures = X.shape
        rng = check_random_state(self.random_state)

        Nk = knn(X,self.k)
        S = affinityGraph(Nk,X)
        L = laplacian(S)

        D = np.identity(nFeatures)
        W  = np.zeros((nFeatures, self.nFeaturesOut), dtype=float)

        G = np.random.randint(0, 2, size=(nSamples, self.nClusters))
        sizes = np.maximum(G.sum(axis=0), 1) #Potencia de matriz
        A = np.diag(1.0 / np.sqrt(sizes)) #Posible al ser una matriz diagonal
        U = G @ A

        gamma = 0.0

        lastScore = 1e300
        rowNorm = None

        noNull = 1e-12

        for _ in range(self.maxIter):
            UUt = sparse.csr_matrix(U @ U.T)
            AUX = L - (self.lambdaArg * (UUt))
            M = (X.T @ AUX @ X) + self.alpha * D
            M = symmetrize(M)
            _, vecs = eigh(M, subset_by_index=[0, self.nFeaturesOut-1])
            W = vecs

            rowNorm = np.linalg.norm(W,axis=1)
            num = rowNorm**2 + noNull
            diagD = 0.5 / np.sqrt(num)
            D = np.diag(diagD)

            Mu = X @ W @ W.T @ X.T
            Mu = symmetrize(Mu)

            _, vecs = eigh(Mu.toarray(), subset_by_index=[Mu.shape[0] - self.nClusters,Mu.shape[0]-1])
            U = vecs
            
            Z = X @ W
            S, gamma = adaptiveAffinityGraph(Z, self.k, self.lambdaArg, eps=noNull)
            L = laplacian(S)

            obj1 = np.sum(Z * (L @ Z))

            row2 = np.sum(W*W, axis=1)  
            obj2 = self.alpha * np.sum(diagD * row2)

            obj3 = self.lambdaArg * (np.linalg.norm(Z.T @ U, 'fro') ** 2)

            obj4 = gamma * np.sum(S * S)

            objFunction = obj1 + obj2 - obj3 + obj4

            if abs(lastScore - objFunction) / (lastScore + noNull) < self.tol:
                break

            lastScore = objFunction

        self.scores_ = rowNorm