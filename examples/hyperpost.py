#######################################
# pyGPGO examples
# hyperpost: shows posterior distribution of hyperparameters
# for a Gaussian Process example
#######################################

import numpy as np
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.covfunc import matern32


if __name__ == '__main__':
    np.random.seed(1337)
    sexp = matern32()
    gp = GaussianProcessMCMC(sexp, niter=2000, init='MAP', step=None)

    X = np.linspace(0, 6, 7)[:, None]
    y = np.sin(X).flatten()
    gp.fit(X, y)
    gp.posteriorPlot()