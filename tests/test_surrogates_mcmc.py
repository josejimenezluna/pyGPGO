import numpy as np
import pymc3 as pm
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.surrogates.tStudentProcessMCMC import tStudentProcessMCMC
from pyGPGO.covfunc import squaredExponential


def test_GP():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp, step=pm.Slice)
    gp.fit(X, y)


def test_tSP():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    sexp = squaredExponential()
    tsp = tStudentProcessMCMC(sexp, step=pm.Slice, niter=100)
    tsp.fit(X, y)


if __name__ == '__main__':
    test_GP()
    test_tSP()
