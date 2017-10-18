#######################################
# pyGPGO examples
# integratedacq: Shows the computation of the integrated acquisition function.
#######################################

import numpy as np
import matplotlib.pyplot as plt
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.GPGO import GPGO

import pymc3 as pm

if __name__ == '__main__':
    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp, step=pm.Slice)

    def f(x):
        return np.sin(x)

    np.random.seed(200)
    param = {'x': ('cont', [0, 6])}
    acq = Acquisition(mode='IntegratedExpectedImprovement')
    gpgo = GPGO(gp, acq, f, param)
    gpgo._firstRun(n_eval=7)

    plt.figure()
    plt.subplot(2, 1, 1)

    Z = np.linspace(0, 6, 100)[:, None]
    post_mean, post_var = gpgo.GP.predict(Z, return_std=True, nsamples=200)
    for i in range(200):
        plt.plot(Z.flatten(), post_mean[i], linewidth=0.4)

    plt.plot(gpgo.GP.X.flatten(), gpgo.GP.y, 'X', label='Sampled data', markersize=10, color='red')
    plt.grid()
    plt.legend()

    xtest = np.linspace(0, 6, 200)[:, np.newaxis]
    a = [-gpgo._acqWrapper(np.atleast_2d(x)) for x in xtest]
    plt.subplot(2, 1, 2)
    plt.plot(xtest, a, label='Integrated Expected Improvement')
    plt.grid()
    plt.legend()
    plt.show()
