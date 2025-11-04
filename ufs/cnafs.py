import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state
from .base import BaseEstimatorUFS 
from .utils.knn import laplacian, symmetrize, pairwise_sq_distances, softmax, randomOrth

class cnafs(BaseEstimatorUFS):
    
    def __init__(self, nFeaturesOut=None, alpha=1.0, beta=100.0, gamma=100.0, lambdaArg=100.0, epsilon=1.0, nClusters=2, k=5, maxIter=100,tol=1e-4,random_state=None):
        super().__init__(nFeaturesOut)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambdaArg = lambdaArg
        self.epsilon = epsilon
        self.nClusters=nClusters
        self.k = k
        self.maxIter = maxIter
        self.tol = tol
        self.random_state = random_state
    
    @staticmethod
    def _buildS(Y_p, V, alpha, beta, gamma, noNull):
        Df = pairwise_sq_distances(Y_p)
        Dv = pairwise_sq_distances(V.T)

        logits = -(alpha*Df + 0.5*gamma*Dv) / (2.0*alpha*beta)

        S = softmax(logits.copy(),noNull)
        S = symmetrize(S)
        return sparse.csr_matrix(S)

    @staticmethod
    def _gpi(A,B, rng, maxIter=5):
        n, c = B.shape

        A = symmetrize(A)

        Y = randomOrth(n, c, rng)

        alpha = float(np.linalg.eigvalsh(A).max())

        if alpha < 0.0:
            alpha = 0.0

        for _ in range(maxIter):
            M = alpha*Y - (A @ Y) + 2.0*B
            U, _, V_t = np.linalg.svd(M, full_matrices=False)
            Y = U @ V_t

        return Y


    def _core(self,X):
        nSamples, nFeatures = X.shape
        rng = check_random_state(self.random_state)

        C_n = np.identity(nSamples) - 1/nSamples * np.ones((nSamples,nSamples))

        Y_p = randomOrth(nSamples,self.nClusters,rng)

        W = np.random.rand(nFeatures, self.nClusters)
        G = np.abs(np.random.rand(nSamples, self.k))
        V = np.abs(np.random.rand(self.k, nSamples))

        Q= np.ones((self.k,self.k)) - np.identity(self.k)

        lastScore = 1e300

        noNull = 1e-12

        for _ in range(self.maxIter):
            S = self._buildS(Y_p, V, self.alpha, self.beta, self.gamma, noNull)
            L,D = laplacian(S, normalized=False, return_degree=True)

            d_w = 0.5 / np.sqrt(np.sum(W*W, axis=1) + self.epsilon) #vector con la diagonal del Lambda
            A = X.T @ C_n @ X #Calculamos la primera parte de la igualdad para calcula W
            A[np.diag_indices_from(A)] += self.lambdaArg * d_w #Añadimos la suma en las diagonales sin materializar la Lambda
            W = np.linalg.solve(A, X.T @ C_n @ Y_p) #Más eficiente que calcular la inversa

            # Cálculo de la matriz Yp
            A = C_n + 2.0*self.alpha*L.toarray()
            B = C_n @ X @ W
            Y_p = self._gpi(A, B, rng)

            # Actualización de G y V
            C = X @ X.T

            numG = C @ V.T
            denG = C @ G @ (V @ V.T)

            np.divide(numG, denG + noNull, out=numG)
            G *= numG

            C = G.T @ C

            numV = C + self.gamma * V @ S
            denV = (C @ G @ V) + (self.gamma * V @ D) + (self.epsilon * Q @ V)

            np.divide(numV, denV + noNull, out=numV)
            V *= numV
            den = np.linalg.norm(V, axis=0)
            den = np.where(den > noNull, den, 1.0)
            V /= den

            # Comprobación de la convergencia y criterio de parada
            obj1 = np.linalg.norm(X.T - X.T@G@V, 'fro')**2
            obj2 = np.linalg.norm(C_n@(X@W-Y_p), 'fro')**2
            obj3 = self.lambdaArg * np.dot(d_w, np.sum(W*W,axis=1)) 
            obj4 = 2 * self.alpha * np.trace(Y_p.T@L@Y_p)
            obj5 = 2 * self.alpha * self.beta * np.sum(S.data * np.log(S.data+noNull))
            obj6 = self.gamma * np.trace(V@L@V.T)
            obj7 = self.epsilon * np.trace(V.T@Q@V)

            objFunction = obj1 + obj2 + obj3 + obj4 + obj5 + obj6 + obj7

            if abs(lastScore - objFunction) / (lastScore + noNull) < self.tol:
                break

            lastScore = objFunction

        self.scores_ = np.linalg.norm(W,axis=1)