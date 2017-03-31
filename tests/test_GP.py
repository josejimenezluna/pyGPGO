import numpy as np
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.covfunc import squaredExponential

def test_GP():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    sexp = squaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    gp.fit(X, y)

    params = gp.getcovparams()

    assert 0.36 < params['l'] < 0.37
    assert 0.39 < params['sigmaf'] < 0.41
    assert 0.29 < params['sigman'] < 0.3

if __name__ == '__main__':
    test_GP()