#######################################
# pyGPGO examples
# example2d: SHows how the Bayesian Optimization works on a two-dimensional
# rastrigin function, step by step.
#######################################


import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential


def rastrigin(x, y, A=10):
    return (2 * A + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y)))


def plot_f(x_values, y_values, f):
    z = np.zeros((len(x_values), len(y_values)))
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            z[i, j] = f(x_values[i], y_values[j])
    plt.imshow(z.T, origin='lower', extent=[np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)])
    plt.colorbar()
    plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'mthesis_text/figures/chapter3/rosen/rosen.pdf'))


def plot2dgpgo(gpgo):
    tested_X = gpgo.GP.X
    n = 100
    r_x, r_y = gpgo.parameter_range[0], gpgo.parameter_range[1]
    x_test = np.linspace(r_x[0], r_x[1], n)
    y_test = np.linspace(r_y[0], r_y[1], n)
    z_hat = np.empty((len(x_test), len(y_test)))
    z_var = np.empty((len(x_test), len(y_test)))
    ac = np.empty((len(x_test), len(y_test)))
    for i in range(len(x_test)):
        for j in range(len(y_test)):
            res = gpgo.GP.predict([x_test[i], y_test[j]])
            z_hat[i, j] = res[0]
            z_var[i, j] = res[1][0]
            ac[i, j] = -gpgo._acqWrapper(np.atleast_1d([x_test[i], y_test[j]]))
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Posterior mean')
    plt.imshow(z_hat.T, origin='lower', extent=[r_x[0], r_x[1], r_y[0], r_y[1]])
    plt.colorbar()
    plt.plot(tested_X[:, 0], tested_X[:, 1], 'wx', markersize=10)
    a = fig.add_subplot(2, 2, 2)
    a.set_title('Posterior variance')
    plt.imshow(z_var.T, origin='lower', extent=[r_x[0], r_x[1], r_y[0], r_y[1]])
    plt.plot(tested_X[:, 0], tested_X[:, 1], 'wx', markersize=10)
    plt.colorbar()
    a = fig.add_subplot(2, 2, 3)
    a.set_title('Acquisition function')
    plt.imshow(ac.T, origin='lower', extent=[r_x[0], r_x[1], r_y[0], r_y[1]])
    plt.colorbar()
    gpgo._optimizeAcq(method='L-BFGS-B', n_start=500)
    plt.plot(gpgo.best[0], gpgo.best[1], 'gx', markersize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'mthesis_text/figures/chapter3/rosen/{}.pdf'.format(item)))
    plt.show()


if __name__ == '__main__':
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    plot_f(x, y, rastrigin)

    np.random.seed(20)
    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')

    param = OrderedDict()
    param['x'] = ('cont', [-1, 1])
    param['y'] = ('cont', [-1, 1])

    gpgo = GPGO(gp, acq, rastrigin, param, n_jobs=-1)
    gpgo._firstRun()

    for item in range(7):
        plot2dgpgo(gpgo)
        gpgo.updateGP()
