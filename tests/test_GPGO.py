import numpy as np
import pymc3 as pm
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.surrogates.RandomForest import RandomForest
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPGO import GPGO


def f(x):
    return -((6 * x - 2) ** 2 * np.sin(12 * x - 4))


def test_GPGO():
    np.random.seed(20)
    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')
    params = {'x': ('cont', (0, 1))}
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter=10)
    res = gpgo.getResult()[0]
    assert .6 < res['x'] < .8


def test_GPGO_mcmc():
    np.random.seed(20)
    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp, step=pm.Slice, niter=100)
    acq = Acquisition(mode='IntegratedExpectedImprovement')
    params = {'x': ('cont', (0, 1))}
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter=10)
    res = gpgo.getResult()[0]
    assert .6 < res['x'] < .8


def test_GPGO_sk():
    np.random.seed(20)
    rf = RandomForest()
    acq = Acquisition(mode='ExpectedImprovement')
    params = {'x': ('cont', (0, 1))}
    gpgo = GPGO(rf, acq, f, params)
    gpgo.run(max_iter=10)
    res = gpgo.getResult()[0]
    assert .7 < res['x'] < .8


if __name__ == '__main__':
    test_GPGO()
    test_GPGO_mcmc()
    test_GPGO_sk()
