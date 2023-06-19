#_____________ 1066600 _________________#
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import scipy.io

#------------------------------------------#
# Load files
def Load():
    # Load .mat file
    mat = scipy.io.loadmat('data32.mat')
    Xi = np.array(mat['stars'])
    Xj = np.array(mat['circles'])

    # Add label
    Yi = np.ones((21,1))
    Yj = -1 * np.ones((21,1))

    # Return Arrays
    return Xi, Xj, Yi, Yj
#------------------------------------------#

#------------------------------------------#
# Gaussian Kernel
def K(X,Y):
    h = 0.08
    Kernel_matrix = np.exp( (-1/h) * ( (X[0] - Y[0])**2 + (X[1] - Y[1])**2 ) )
    return Kernel_matrix
#------------------------------------------#

#------------------------------------------#
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

#------------------------------------------#

#------------------------------------------#
# Representer theorem
def F_hat(X, Xi, Xj, a, b):
    Stars_term = np.zeros((21,1))
    Circles_term = np.zeros((21,1))
    for i in range(0,21):
        Stars_term[i] = a[i] * K(X, Xi[i])
        Circles_term[i] = b[i] * K(X, Xj[i])
    return np.sum(Stars_term) + np.sum(Circles_term)
#------------------------------------------#

#------------------------------------------#
def create_Grid():
    # Generate a grid of points
    Grid = []
    x_cor = np.linspace(-1.1, 1.2,300)
    y_cor = np.linspace(-0.2, 1.2,300)
    for x in x_cor:
        for y in y_cor:
            Grid.append([x , y])
    Grid = np.array(Grid)
    return Grid
#------------------------------------------#

#---------- Main ------------#
# Define hyperparameter
lamda = 0.2

[Xi, Xj, Yi, Yj] = Load()
[a, b] = train(Xi, Xj, Yi, Yj, lamda)
Grid = create_Grid()

# Find boundary
boundary = []
for spot in Grid:
    RT = F_hat(spot, Xi, Xj, a, b)
    if RT < 0.01 and RT > -0.01:
        boundary.append(spot)

boundary = np.array(boundary)
# Plot boundary
plt.plot(boundary[:,0],boundary[:,1],'.',color = 'r')

# Plot Stars
plt.plot(Xi[:,0],Xi[:,1],'o',color = 'b')

# Plot Circles
plt.plot(Xj[:,0],Xj[:,1],'o',color = 'g')
plt.show()

