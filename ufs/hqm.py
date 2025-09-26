import numpy as np
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

    def _core(self, X):
        nSamples, nFeatures = X.shape
        rng = check_random_state(self.random_state)

        obs = ~np.isnan(X)                # Comprueba los que son NaN
        D   = obs.astype(float)           # Pone a 1 los valores presentes y a 0 los Nan
        Xf  = np.where(obs, X, 0.0)       # Sustituye los NaN por 0 para evitar posibles fallos

        #A continuación inicializamos la W con la regresión Ridge
        W = np.zeros((nFeatures,nFeatures),dtype=float)

        for j in range(nFeatures):
            y = Xf[:,j] #Tomamos la columna j de X sin los NaN
            sw = D[:,j] #Esto va a servir para que se tomen en cuenta solo los que se han observado

            if sw.sum() == 0:   #En caso de que no haya ninguna observada en la columna no hace falta enternar nada
                continue

            reg = Ridge(alpha=self.lambdaArg, fit_intercept=False)
            reg.fit(Xf, y, sample_weight=sw) #Ajusta solo los valores que se han observado

            wj = reg.coef_
            wj[j] = 0.0 #Para que una columna j se explique a sí misma a sí misma.
            W[:,j] = wj

        R = Xf - Xf @ W                 ##
        Fi = ((D * R) ** 2).sum(axis=1) ## Inicializamos la f tilda y las Fi que las vamos a necesitar
        ftilde = Fi.mean()              ##

        mu = 1.0

        lastScore = 1e300
        rowNorm = np.linalg.norm(W,axis=1)
        noNull = 1e-12

        for _ in range(self.maxIter):
            v = (mu/(mu+Fi+noNull))**2

            Q = np.diag(1.0/(2.0*rowNorm+noNull))

            sv = np.sqrt(v)
            Xtilda = Xf * sv[:,None]
            B = D * Xtilda
            Aux = B.T @ Xtilda
            A = Aux + self.lambdaArg * Q
            W = np.linalg.solve(A,Aux.T) #Calcula la inversa de una forma más eficiente

            R = Xf - Xf @ W                 
            Fi = ((D * R) ** 2).sum(axis=1) 
            ftilde = Fi.mean()

            mu = np.max((mu*0.5,ftilde*0.5))

            #Ahora vamos a calcular si es necesario parar en esta iteración debido a que no se ha mejorado lo suficiente
            obj1 = (v * Fi).sum()
            obj2 = (mu * (sv - 1)**2).sum()
            rowNorm = np.linalg.norm(W,axis=1)
            obj3 = self.lambdaArg * rowNorm.sum()

            objFunction = obj1 + obj2 + obj3

            if abs(lastScore - objFunction) / (lastScore + noNull) < self.tol:
                break

            lastScore = objFunction

        self.scores_ = rowNorm






        
            





