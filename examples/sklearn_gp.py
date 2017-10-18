#######################################
# pyGPGO examples
# sklearn_gp: Replicates sklearn example for hyperparam opt.
#######################################

import numpy as np
import matplotlib.pyplot as plt
from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

    sexp = squaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    gp.fit(X, y)

    X_ = np.linspace(0, 5, 100)
    y_mean, y_var = gp.predict(X_[:, np.newaxis], return_std=True)
    y_std = np.sqrt(y_var)
    plt.plot(X_, y_mean, 'k', lw=2, zorder=9, label='Posterior mean')
    plt.fill_between(X_, y_mean - 1.64 * y_std, y_mean + 1.64 * y_std, alpha=0.4, color='blue')
    plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=2, zorder=9, label='Original function')
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10)
    plt.legend(loc=0)
    params = gp.getcovparams()
    plt.tight_layout()
    plt.show()
    