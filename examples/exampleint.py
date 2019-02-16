#######################################
# pyGPGO examples
# exampleint: tests and visualizes an integrated acquisition function.
#######################################

import matplotlib.pyplot as plt

import numpy as np
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
import pymc3 as pm


def plotGPGO(gpgo, param):
    param_value = list(param.values())[0][1]
    x_test = np.linspace(param_value[0], param_value[1], 1000).reshape((1000, 1))
    fig = plt.figure()
    a = np.array([-gpgo._acqWrapper(np.atleast_1d(x)) for x in x_test]).flatten()
    r = fig.add_subplot(1, 1, 1)
    r.set_title('Acquisition function')
    plt.plot(x_test, a, color='green')
    gpgo._optimizeAcq(method='L-BFGS-B', n_start=25)
    plt.axvline(x=gpgo.best, color='black', label='Found optima')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    np.random.seed(321)

    def f(x):
        return (np.sin(x))

    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp, step=pm.Slice)
    acq = Acquisition(mode='IntegratedExpectedImprovement')
    param = {'x': ('cont', [0, 2 * np.pi])}

    gpgo = GPGO(gp, acq, f, param, n_jobs=-1)
    gpgo._firstRun()

    for i in range(6):
        plotGPGO(gpgo, param)
        gpgo.updateGP()
