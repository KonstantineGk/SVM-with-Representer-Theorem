import numpy as np

# Gaussian Kernel
def K(X,Y):
    h = 0.08
    Kernel_matrix = np.exp( (-1/h) * ( (X[0] - Y[0])**2 + (X[1] - Y[1])**2 ) )
    return Kernel_matrix