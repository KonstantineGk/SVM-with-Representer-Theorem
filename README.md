# SVM-Implementation-with-Gaussian-Kernel-and-Representer-Theorem
Implemented a Support Vector Machine (SVM) algorithm with a Gaussian kernel to establish an effective decision boundary between two datasets.
The algorithm utilized the Representer Theorem, resulting in improved computational efficiency.

This project was completed from scratch for the Machine Learning course( University of Patras ).

1) First you need to install NumPy, Math, MatPlotLib and SciPy( pip install ...).
2) Put the "data32.mat" in the same file as the script or modify for your own dataset.
3) Run :D

FUNCTIONS:
1) Load: Loads the .mat file and returns seperate np.arrays for both data and labes.
2) K: Implementation of a Gaussian Kernel.
3) train: Calculate the a,b weights for the RT using the Gaussian Kernel.(Wikipedia: Representer Theorem)
4) F_hat: Calculate Loss linearly for an item using the RT.
5) create_Grid: Create a 2D grid for plotting.
6) main: Calculate Loss for each spot in Grid. Then is Loss close to 0 then spot is part of Boundary.
   Finally Plot with seperate colors.

Thank You!
