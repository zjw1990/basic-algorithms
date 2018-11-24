import numpy as np 
from scipy import linalg

#QR solution of CCA, same as the solution of matlab
def CCA(x,y, dim = 0):
    shapex = np.shape(x)
    shapey = np.shape(y)
    n = shapex[0]
    if shapey[0] != n:
        print("error, sample size should be same")
        return 0,0,0,0,0
    p1 = shapex[1]
    p2 = shapey[1]


    #meancentering
    x = mean_center(x)
    y = mean_center(y)

    #qr, permulation version

    q11,t11,perm1 = linalg.qr(x,pivoting=True, mode = "economic")
    q22,t22,perm2 = linalg.qr(y,pivoting=True, mode = "economic")

    #rank calculation
    rankx = np.linalg.matrix_rank(x)
    ranky = np.linalg.matrix_rank(y)

    # CCA
    dim = dim
    l,d,m_t = np.linalg.svd(np.matmul(q11.T,q22), full_matrices=False)
    m = m_t.T

    A_permu = np.matmul(linalg.inv(t11),l[:,:dim])*np.sqrt(n-1)
    B_permu = np.matmul(linalg.inv(t22),m[:,:dim])*np.sqrt(n-1)
    
    # repermuation
    A = repermuation(A_permu,perm1)
    B = repermuation(B_permu,perm2)

 

    return A,B,d[:dim]


def repermuation(x,permutaion):
    A = np.zeros(np.shape(x))
    for i in range(len(x)):
        A[permutaion[i]] = x[i]
    return A

def mean_center(x):
    x = np.array(x)
    avg = np.mean(x, axis=0)
    x -= avg
    return x