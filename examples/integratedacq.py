import numpy as np
import matplotlib.pyplot as plt
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.GPGO import GPGO


if __name__ == '__main__':
    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp)
    # gp.fit(X, y)
    # Z = np.linspace(0, 6, 100)[:, None]
    # post_mean, post_var = gp.predict(Z, return_std=True, nsamples=200)
    #
    #
    # import matplotlib.pyplot as plt
    # for i in range(100):
    #     plt.plot(Z.flatten(), post_mean[i])
    # plt.show()
    #
    def f(x):
        return np.sin(x)

    np.random.seed(200)
    param = {'x': ('cont', [0, 6])}
    acq = Acquisition(mode='IntegratedExpectedImprovement')
    gpgo = GPGO(gp, acq, f, param)
    gpgo._firstRun(n_eval=7)

    plt.figure()
    plt.plot(gpgo.GP.X.flatten(), gpgo.GP.y, '.')

    Z = np.linspace(0, 6, 100)[:, None]
    post_mean, post_var = gpgo.GP.predict(Z, return_std=True, nsamples=200)
    plt.figure()
    for i in range(200):
        plt.plot(Z.flatten(), post_mean[i])


    xtest = np.linspace(0, 6, 50)[:, np.newaxis]
    a = [-gpgo._acqWrapper(np.atleast_2d(x)) for x in xtest]
    plt.figure()
    plt.plot(xtest, a)