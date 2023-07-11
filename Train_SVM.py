import numpy as np
from Gaussian_Kernel import K 

# Find optimal weights
def train(Xi, Xj, Yi, Yj, lamda):
    # Joined X
    X = np.concatenate((Xi,Xj), axis = 0)
    # Joined Y
    Y= np.concatenate((Yi,Yj) , axis = 0)
    # Start A
    A = np.zeros((42,42))
    
    # Fill A with Aij = K(X[i],X[j])
    for i in range(0,42):
        for j in  range(0,42):
            A[i][j] = K(X[i], X[j])
            
    # Calculate Weights
    ab = np.dot( np.dot( np.linalg.inv( np.dot(A.T,A) + lamda*A), A ) , Y )
    a = ab[:21]
    b = ab[21:]
    
    # Return a and b
    return a, b