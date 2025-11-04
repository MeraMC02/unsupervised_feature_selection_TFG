import numpy as np
from scipy.linalg import eigh
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .base import BaseEstimatorUFS 
from .utils.knn import adaptiveAffinityGraph, laplacian, symmetrize, randomOrth


class rsogfs(BaseEstimatorUFS):

    def __init__(self, nFeaturesOut=None, nClusters=2, k=5, maxIter=30,tol=1e-4,random_state=None):
        super().__init__(nFeaturesOut)
        self.nClusters=nClusters
        self.k = k
        self.maxIter = maxIter
        self.tol = tol
        self.random_state = random_state

    def _core(self, X):
        nSamples, nFeatures = X.shape
        rng = check_random_state(self.random_state)
        noNull = 1e-12

        pca = PCA(n_components=self.nClusters, random_state=self.random_state).fit(X)
        W0 = pca.components_.T     ## 
        U,_,Vt = np.linalg.svd(W0, full_matrices=False)                                         ## Mayor estabilidad que random
        W = U @ Vt                                                                              ## 

        S, _ = adaptiveAffinityGraph(X,self.k,1.0,noNull)
        L = laplacian(S)

        lastScore = 1e300
        for _ in range(self.maxIter):
            Aux = (X.T @ (L @ X))
            Aux = symmetrize(Aux)
            lambdaArg = np.linalg.eigvalsh(Aux).max()
            A = lambdaArg*np.identity(nFeatures) - Aux
            A = symmetrize(A)

            AW = A @ W
            M = W.T @ AW
            tau = 1e-4 * np.trace(M) / M.shape[0]    # escala-invariante
            Minv = np.linalg.pinv(M + tau * np.identity(self.nClusters))       # Moore–Penrose estable
            T = AW @ Minv                                      
            diagP = np.sum(T * AW, axis=1) 

            #Selección parcial: contiene exactamente las k mayores (orden arbitrario)
            I_part = np.argpartition(diagP, -self.nFeaturesOut)[-self.nFeaturesOut:]
            I = np.sort(I_part)

            Atil = A[np.ix_(I,I)]
            Atil = symmetrize(Atil) #Ya es simétrica pero evitamos error de cómputo

            U = np.zeros((nFeatures,self.nFeaturesOut), dtype=float)
            U[I,np.arange(self.nFeaturesOut)] = 1.0

            _, vecs = eigh(Atil, subset_by_index=[Atil.shape[0] - self.nClusters,Atil.shape[0]-1])
            V = vecs

            W = U @ V

            Z = X @ W
            Z = StandardScaler().fit_transform(Z) #Estandarizamos porque puede mejorar el algoritmo
            S, gamma = adaptiveAffinityGraph(Z,self.k,1.0,noNull)
            L = laplacian(S)

            obj1 = float(np.sum(Z * (L @ Z))) #Tr(Z.T@L@Z)
            obj2 = gamma * float(S.multiply(S).sum())
            objFunction = obj1 + obj2

            if abs(lastScore - objFunction) / (lastScore + noNull) < self.tol:
                break

            lastScore = objFunction

        self.scores_ = np.linalg.norm(W,axis=1)



       

    




