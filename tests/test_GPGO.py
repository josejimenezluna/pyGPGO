import numpy as np
from pyGPGO.covfunc import squaredExponential
from pyGPGO.GPRegressor import GPRegressor
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPGO import GPGO

def f(x):
    return -((6 * x - 2) ** 2 * np.sin(12 * x - 4))

def test_GPGO():
    np.random.seed(20)
    sexp = squaredExponential()
    gp = GPRegressor(sexp)
    acq = Acquisition(mode= 'ExpectedImprovement')
    params = {'x': ('cont', (0, 1))}
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter=10)
    res = gpgo.getResult()[0]
    assert .7 < res['x'] < .8

if __name__ == '__main__':
    test_GPGO()