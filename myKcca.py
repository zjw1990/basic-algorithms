import numpy as np 
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from cca import CCA
from kcca import KCCA

def myKCCA(x, y, kernel, dim, degree,epsilon, gamma,coef0,n_jobs):

    kx = pairwise_kernels(x, Y=None, metric=kernel, filter_params=True, n_jobs=n_jobs, degree=degree, gamma=gamma, coef0=coef0)
    ky = pairwise_kernels(y, Y=None, metric=kernel, filter_params=True, n_jobs=n_jobs, degree=degree, gamma=gamma, coef0=coef0)
    print("my kx is",kx[0][:3])
    wx,wy,r = CCA(kx,ky, dim=dim)

    return wx,wy,r



	
X = np.random.normal(size=(1000,100))
Y = np.random.normal(size=(1000,20))
kcca = KCCA(n_components=10, kernel="rbf", n_jobs=1, epsilon=0.1).fit(X, Y)
alpha = kcca.alpha
beta = kcca.beta
X_test = np.random.normal(size=(10,100))
Y_test = np.random.normal(size=(10,20))
Kx = kcca._pairwise_kernels(X_test, X)
Ky = kcca._pairwise_kernels(Y_test, Y)
F = np.dot(Kx, alpha)
G = np.dot(Ky, beta)
print(F[0][:3])

wx,wy,r = myKCCA(X,Y,kernel="rbf",dim=10,degree=3,epsilon=0.1,gamma=None,coef0=1,n_jobs=1)
Fme = np.dot(Kx, wx)
print(Fme[0][:3])
print(r)
