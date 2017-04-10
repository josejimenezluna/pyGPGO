import numpy as np
import matplotlib.pyplot as plt
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.covfunc import squaredExponential


if __name__ == '__main__':
    np.random.seed(1337)
    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp, niter=2000, init='MAP')

    X = np.linspace(0, 6, 7)[:, None]
    y = np.sin(X).flatten()
    gp.fit(X, y)
    gp.posteriorPlot()