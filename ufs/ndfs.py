import numpy as np
from scipy.linalg import eigh
from sklearn.utils import check_random_state
from .base import BaseEstimatorUFS 
from .utils.knn import affinityGraph, knn, laplacian

class ndfs(BaseEstimatorUFS):

    def __init__(self, nFeaturesOut=None, nClusters=2, k=5, alpha=1.0, beta=1.0, gamma=1e8,maxIter=30,tol=1e-4,random_state=None):
        super().__init__(nFeaturesOut)
        self.nClusters=nClusters
        self.k=k
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.maxIter=maxIter
        self.tol=tol
        self.random_state=random_state

    def _core(self,X):
        nSamples, nFeatures = X.shape
        rng = check_random_state(self.random_state)

        Nk = knn(X,self.k)
        S = affinityGraph(Nk,X)
        L = laplacian(S, normalized=True)

        vals, vecs = eigh(L.toarray(), subset_by_index=[0, self.nClusters-1])
        F = np.maximum(vecs, 0)
        F /= np.linalg.norm(F, axis=0)

        D = np.identity(nFeatures)
        W  = np.zeros((nFeatures, self.nClusters), dtype=float)

        lastScore = 1e300
        rowNorm = None

        noNull = 1e-12

        for _ in range(self.maxIter):
            A_inv = np.linalg.inv(X.T @ X + self.beta * D)
            M = L.toarray() + self.alpha * (np.identity(nSamples) - X @ A_inv @ X.T)

            num = self.gamma * F
            denom = M @ F + self.gamma * (F @ F.T @ F) 
            F = F * (num/(denom+noNull))
            F /= np.linalg.norm(F,axis=0)

            W = A_inv @ X.T @ F

            rowNorm = np.linalg.norm(W,axis=1) + noNull
            D = np.diag(0.5 / rowNorm)

            obj1 = np.trace(F.T @ L @ F)
            obj2 = self.alpha * (np.linalg.norm(X@W - F, "fro")**2 + self.beta*np.sum(rowNorm))
            obj3 = 0.5 * self.gamma * (np.linalg.norm(F.T @ F - np.identity(self.nClusters),"fro")**2)
            objFunction = obj1 + obj2 + obj3

            if abs(lastScore - objFunction) / (lastScore + noNull) < self.tol:
                break

            lastScore = objFunction

        self.scores_ = rowNorm