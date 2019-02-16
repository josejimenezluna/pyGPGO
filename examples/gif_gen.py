#######################################
# pyGPGO examples
# gif_gen: generates a gif (the one in paper.md) showing how the BO framework
# works on the Franke function, step by step.
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
    x = np.linspace(0, 1, num=1000)
    y = np.linspace(0, 1, num=1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('Original function')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)


def plotPred(gpgo, num=100):
    X = np.linspace(0, 1, num=num)
    Y = np.linspace(0, 1, num=num)
    U = np.zeros((num**2, 2))
    i = 0
    for x in X:
        for y in Y:
            U[i, :] = [x, y]
            i += 1
    z = gpgo.GP.predict(U)[0]
    Z = z.reshape((num, num))
    X, Y = np.meshgrid(X, Y)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title('Gaussian Process surrogate')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    best = gpgo.best
    ax.scatter([best[0]], [best[1]], s=40, marker='x', c='r', label='Sampled point')
    plt.legend(loc='lower right')
    #plt.show()
    return Z


if __name__ == '__main__':
    n_iter = 10
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {'x': ('cont', [0, 1]),
             'y': ('cont', [0, 1])}

    np.random.seed(85)
    gpgo = GPGO(gp, acq, f, param)
    gpgo.run(max_iter=1)

    for i in range(n_iter):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle("Franke's function (Iteration {})".format(i+1))
        gpgo.run(max_iter=1, resume=True)
        plotFranke()
        plotPred(gpgo)
        plt.show()
        #plt.savefig('/home/jose/gif/{}.png'.format(i), dpi=300)
        plt.close()
