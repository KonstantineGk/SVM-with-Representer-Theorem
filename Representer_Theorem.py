import numpy as np
from Gaussian_Kernel import K

# Representer theorem
def F_hat(X, Xi, Xj, a, b):
    Stars_term = np.zeros((21,1))
    Circles_term = np.zeros((21,1))
    for i in range(0,21):
        Stars_term[i] = a[i] * K(X, Xi[i])
        Circles_term[i] = b[i] * K(X, Xj[i])
    return np.sum(Stars_term) + np.sum(Circles_term)