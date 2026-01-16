import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge
from .base import BaseEstimatorUFS 

class hqm(BaseEstimatorUFS):
    
    def __init__(self, nFeaturesOut=None, lambdaArg=100.0, nClusters=2, k=5, maxIter=30,tol=1e-4,random_state=None):
        super().__init__(nFeaturesOut)
        self.lambdaArg = lambdaArg
        self.nClusters=nClusters
        self.k = k
        self.maxIter = maxIter
        self.tol = tol
        self.random_state = random_state
    
    def _ridge_one_col(self, j, Xf, D, solver):
        y = Xf[:, j]
        sw = D[:, j]

        if sw.sum() == 0:
            return j, None

        reg = Ridge(alpha=self.lambdaArg, fit_intercept=False, solver=solver)
        reg.fit(Xf, y, sample_weight=sw)

        wj = reg.coef_.copy()
        wj[j] = 0.0
        return j, wj
    
    def _init_W_ridge_parallel(self, Xf, D):
        nSamples, nFeatures = Xf.shape

        # REGLA SIMPLE PEDIDA
        solver = "lsqr" if nFeatures > 1000 else "auto"

        results = Parallel(
            n_jobs=-1,
            prefer="threads"   # opción más robusta en general
        )(
            delayed(self._ridge_one_col)(j, Xf, D, solver)
            for j in range(nFeatures)
        )

        W = np.zeros((nFeatures, nFeatures), dtype=float)
        for j, wj in results:
            if wj is not None:
                W[:, j] = wj
        return W

    def _hqm_step(self, Xf, D, W, Fi, rowNorm, mu, noNull):
       nFeatures = W.shape[0]

       v = (mu / (mu + Fi + noNull)) ** 2
       sv = np.sqrt(v)

       Xtilda = Xf * sv[:, None]
       B = D * Xtilda
       Aux = B.T @ Xtilda

       Qdiag = 1.0 / (2.0 * rowNorm + noNull)

       A = Aux.copy()
       A.flat[::nFeatures + 1] += self.lambdaArg * Qdiag

       W_new = np.linalg.solve(A, Aux.T)

       R_new = Xf - Xf @ W_new
       Fi_new = ((D * R_new) ** 2).sum(axis=1)

       mu_new = max(mu * 0.5, Fi_new.mean() * 0.5)
       rowNorm_new = np.linalg.norm(W_new, axis=1)

       obj1 = (v * Fi_new).sum()
       obj2 = (mu_new * (sv - 1) ** 2).sum()
       obj3 = self.lambdaArg * rowNorm_new.sum()

       obj = obj1 + obj2 + obj3

       return W_new, Fi_new, rowNorm_new, mu_new, obj

    def _core(self, X):
        nSamples, nFeatures = X.shape
        rng = check_random_state(self.random_state)

        obs = ~np.isnan(X)                # Comprueba los que son NaN
        D   = obs.astype(float)           # Pone a 1 los valores presentes y a 0 los Nan
        Xf  = np.where(obs, X, 0.0)       # Sustituye los NaN por 0 para evitar posibles fallos

        noNull = 1e-12

        #Hacemos una iteración con W=0 y si converge bien, mantenemos dicha inicialización
        W = np.zeros((nFeatures,nFeatures),dtype=float)
        R = Xf - Xf @ W
        Fi = ((D * R) ** 2).sum(axis=1)
        mu = 1.0
        rowNorm = np.linalg.norm(W, axis=1)

        lastScore = 1e300

        W1, Fi1, rowNorm1, mu1, obj1 = self._hqm_step(Xf, D, W, Fi, rowNorm, mu, noNull)

        rel_improve = (Fi.mean() - Fi1.mean()) / (abs(Fi.mean()) + noNull)

        if rel_improve > 1e-3:
            # W=0 funciona bien
            W, Fi, rowNorm, mu = W1, Fi1, rowNorm1, mu1
            lastScore = obj1
            start_iter = 1
        else:
            # Usamos Ridge
            W = self._init_W_ridge_parallel(Xf, D)
            R = Xf - Xf @ W
            Fi = ((D * R) ** 2).sum(axis=1)
            rowNorm = np.linalg.norm(W, axis=1)
            mu = 1.0
            start_iter = 0

        for _ in range(start_iter, self.maxIter):
            W, Fi, rowNorm, mu, obj = self._hqm_step(Xf, D, W, Fi, rowNorm, mu, noNull)

            if abs(lastScore - obj) / (lastScore + noNull) < self.tol:
                break
            lastScore = obj

        self.scores_ = rowNorm






        
            





