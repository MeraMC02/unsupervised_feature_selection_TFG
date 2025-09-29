import numpy as np
from scipy.linalg import eigh
from sklearn.utils import check_random_state
from .base import BaseEstimatorUFS 

class spcafs(BaseEstimatorUFS):
    
    def __init__(self, nFeaturesOut=None, reducedDim=3, p=0.5, gamma=1.0, maxIter=30, noNull=1e-12, tol=1e-4, random_state=None):
        super().__init__(nFeaturesOut)
        self.reducedDim = reducedDim
        self.p = p
        self.gamma = gamma
        self.maxIter = maxIter
        self.noNull = noNull
        self.tol = tol
        self.random_state = random_state

    def _core(self,X):
        nSamples, nFeatures = X.shape

        vect1 = np.ones((nSamples,1),dtype=float)
        H = np.full((nSamples,nSamples), -1.0/nSamples,dtype=X.dtype)
        np.fill_diagonal(H, 1.0 - 1.0/nSamples)
        S = X.T @ (H @ X)

        G = np.identity(nFeatures)

        lastScore = 1e300
        rowNorm = None

        for _ in range(self.maxIter):

            A = self.gamma*G - S
            _, vecs = eigh(A, subset_by_index=[0, self.reducedDim-1])
            W = vecs

            rowNorm = np.linalg.norm(W,axis=1)
            d = (self.p/2.0)*(rowNorm*rowNorm+self.noNull)**((self.p-2)/2.0)
            G = np.diag(d)

            obj1 = np.linalg.trace(W.T@S@W)
            obj2 = self.gamma*np.sum(rowNorm**self.p)

            objFunction = -obj1 + obj2

            if abs(lastScore - objFunction) / (lastScore + self.noNull) < self.tol:
                break

            lastScore = objFunction

        self.scores_ = rowNorm








