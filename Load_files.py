import scipy.io
import numpy as np

# Load files
def Load():
    # Load .mat 
    mat = scipy.io.loadmat(r"C:\Users\Eygenia\Desktop\SVM\data32.mat")
    Xi = np.array(mat['stars'])
    Xj = np.array(mat['circles'])
    
    # Add label
    Yi = np.ones((21,1))
    Yj = -1 * np.ones((21,1))

    # Return Arrays
    return Xi, Xj, Yi, Yj