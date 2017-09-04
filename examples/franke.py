import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

from mpl_toolkits.mplot3d import Axes3D
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

    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {'x': ('cont', [0, 1]),
             'y': ('cont', [0, 1])}

    np.random.seed(1337)
    gpgo = GPGO(gp, acq, f, param)
    gpgo.run(max_iter=10)