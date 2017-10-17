import numpy as np
from pyGPGO.covfunc import squaredExponential, matern, matern32, matern52, \
                           gammaExponential, rationalQuadratic, expSine, dotProd


covfuncs = [squaredExponential(), matern(), matern32(), matern52(), gammaExponential(),
            rationalQuadratic(), expSine(), dotProd()]

grad_enabled = [squaredExponential(), matern32(), matern52(), gammaExponential(),
                rationalQuadratic(), expSine()]

# Some kernels do not have gradient computation enabled, such is the case 
# of the generalised matérn kernel.
#
# All (but the dotProd kernel) have a characteristic length-scale l that
# we test for here.


def test_sim():
    rng = np.random.RandomState(0)
    X = np.random.randn(100, 3)
    for cov in covfuncs:
        cov.K(X, X)


def test_grad(): 
    rng = np.random.RandomState(0)
    X = np.random.randn(3, 3)
    for cov in grad_enabled:
        cov.gradK(X, X, 'l')


if __name__ == '__main__':
    test_sim()
    test_grad()