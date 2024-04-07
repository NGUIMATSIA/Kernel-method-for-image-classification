import numpy as np

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self, X, Y):
        XX = np.sum(X**2, axis=1)[:, np.newaxis]
        YY = np.sum(Y**2, axis=1)[np.newaxis, :]
        distances = XX + YY - 2 * X.dot(Y.T)
        return np.exp(-distances / (2 * self.sigma**2))
    
class Linear:
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return  X@Y.T
    
class Polynomial:
    def __init__(self, d = 100, cst = 0):
        self.d = d  
        self.cst = cst 
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return np.power(X @ Y.T + self.cst,self.d)

class Chi2Kernel:
    def __init__(self, gamma):
        self.gamma = gamma  ## paramètre gamma

    def kernel(self, X1, X2):
        n_X1, n_X2 = X1.shape[0], X2.shape[0]
        out = np.zeros((n_X1, n_X2))

        for i in range(n_X1):
            for j in range(n_X2):
                denominateur = (X1[i] - X2[j])
                nominateur = (X1[i] + X2[j] + 1e-10)
                p = np.sum((denominateur**2) / nominateur)
                out[i, j] = -p

        return np.exp(self.gamma * out)

class Chi2_sum_Gaussian:
    def __init__(self, gamma, sigma):
        self.gamma = gamma
        self.sigma = sigma

    def kernel(self, X1, X2):
        n_X1, n_X2 = X1.shape[0], X2.shape[0]
        out = np.zeros((n_X1, n_X2))
        
        # Chi2 kernel
        for i in range(n_X1):
            for j in range(n_X2):
                denominateur = (X1[i] - X2[j])
                nominateur = (X1[i] + X2[j] + 1e-10)
                p = np.sum((denominateur**2) / nominateur)
                out[i, j] = -p

        # Gaussian kernel 
        XX = np.sum(X1**2, axis=1)[:, np.newaxis]
        YY = np.sum(X2**2, axis=1)[np.newaxis, :]
        distances = XX + YY - 2 * X1.dot(X2.T)

        return np.exp(-distances / (2 * self.sigma**2)) + np.exp(self.gamma * out)



class Chi2_prod_Gaussian:

    def __init__(self, gamma, sigma):
        self.gamma = gamma
        self.sigma = sigma

    def kernel(self, X1, X2):
        n_X1, n_X2 = X1.shape[0], X2.shape[0]
        out = np.zeros((n_X1, n_X2))
        
        # Chi2 kernel
        for i in range(n_X1):
            for j in range(n_X2):
                denominateur = (X1[i] - X2[j])
                nominateur = (X1[i] + X2[j] + 1e-10)
                p = np.sum((denominateur**2) / nominateur)
                out[i, j] = -p

        # Gaussian kernel 
        XX = np.sum(X1**2, axis=1)[:, np.newaxis]
        YY = np.sum(X2**2, axis=1)[np.newaxis, :]
        distances = XX + YY - 2 * X1.dot(X2.T)

        return np.exp(-distances / (2 * self.sigma**2)) * np.exp(self.gamma * out)

