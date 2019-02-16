#######################################
# pyGPGO examples
# franke: optimizes Franke's function.
#######################################

import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

import matplotlib.pyplot as plt
from matplotlib import cm


def f(x, y):
    # Franke's function (https://www.mathworks.com/help/curvefit/franke.html)
    one = 0.75 * np.exp(-(9 * x - 2)**2/4 - (9 * y - 2)**2/4)
    two = 0.75 * np.exp(-(9 * x + 1)**2/49 - (9 * y + 1)/10)
    three = 0.5 * np.exp(-(9 * x - 7)**2/4 - (9*y - 3)**2/4)
    four = 0.25 * np.exp(-(9 * x - 4)**2 - (9*y - 7)**2)
    return one + two + three - four


def plotFranke():
    """
    Plots Franke's function
    """
    x = np.linspace(0, 1, num=1000)
    y = np.linspace(0, 1, num=1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    plotFranke()

    cov = matern32()     # Using a matern v=3/2 covariance kernel
    gp = GaussianProcess(cov)   # A Gaussian Process regressor without hyperparameter optimization
    acq = Acquisition(mode='ExpectedImprovement')   # Expected Improvement acquisition function
    param = {'x': ('cont', [0, 1]),
             'y': ('cont', [0, 1])}     # Specify parameter space

    np.random.seed(1337)
    gpgo = GPGO(gp, acq, f, param)  # Call GPGO class 
    gpgo.run(max_iter=10)   # 10 iterations
    gpgo.getResult()    # Get your result
