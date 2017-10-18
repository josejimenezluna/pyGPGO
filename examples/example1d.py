#######################################
# pyGPGO examples
# example1d: SHows how the Bayesian Optimization works on a one-dimensional
# sine-like function, step by step.
#######################################


import os

import matplotlib.pyplot as plt

import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential


def plotGPGO(gpgo, param):
    param_value = list(param.values())[0][1]
    x_test = np.linspace(param_value[0], param_value[1], 1000).reshape((1000, 1))
    hat = gpgo.GP.predict(x_test, return_std=True)
    y_hat, y_std = hat[0], np.sqrt(hat[1])
    l, u = y_hat - 1.96 * y_std, y_hat + 1.96 * y_std
    fig = plt.figure()
    r = fig.add_subplot(2, 1, 1)
    r.set_title('Fitted Gaussian process')
    plt.fill_between(x_test.flatten(), l, u, alpha=0.2)
    plt.plot(x_test.flatten(), y_hat, color='red', label='Posterior mean')
    plt.legend(loc=0)
    a = np.array([-gpgo._acqWrapper(np.atleast_1d(x)) for x in x_test]).flatten()
    r = fig.add_subplot(2, 1, 2)
    r.set_title('Acquisition function')
    plt.plot(x_test, a, color='green')
    gpgo._optimizeAcq(method='L-BFGS-B', n_start=1000)
    plt.axvline(x=gpgo.best, color='black', label='Found optima')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'mthesis_text/figures/chapter3/sine/{}.pdf'.format(i)))
    plt.show()


if __name__ == '__main__':
    np.random.seed(321)

    def f(x):
        return (np.sin(x))

    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {'x': ('cont', [0, 2 * np.pi])}

    gpgo = GPGO(gp, acq, f, param, n_jobs=-1)
    gpgo._firstRun()

    for i in range(6):
        plotGPGO(gpgo, param)
        gpgo.updateGP()
