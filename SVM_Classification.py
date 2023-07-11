#_____________ 1066600 _________________#
import numpy as np
import matplotlib.pyplot as plt

#----- Import Functions -----#
from Gaussian_Kernel import K
from Representer_Theorem import F_hat
from Grid import create_Grid
from Load_files import Load
from Train_SVM import train

def main():
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

if __name__ == "__main__":
    main()
