import numpy as np

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