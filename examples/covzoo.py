#######################################
# pyGPGO examples
# bayoptwork: Generates a plot to show how the Bayesian Optimization framework
# works, ignoring areas with either low posterior mean or low variance. 
#######################################

import numpy as np
from pyGPGO.covfunc import squaredExponential, matern32, gammaExponential, rationalQuadratic
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Build synthetic data (sine function)
    x = np.arange(0, 2 * np.pi + 0.01, step=np.pi / 2.05)
    y = np.sin(x)
    X = np.array([np.atleast_2d(u) for u in x])[:, 0]

    # Covariance functions to loop over
    covfuncs = [squaredExponential(), matern32(), gammaExponential(), rationalQuadratic()]
    titles = [r'Squared Exponential ($l = 1$)', r'Matérn ($\nu = 1.5$, $l = 1$)',
              r'Gamma Exponential ($\gamma = 1, l = 1$)', r'Rational Quadratic ($\alpha = 1, l = 1$)']

    cm_bright = ['#9ad2cb', '#add0cd', '#b8c3ce', '#9daec9']
    #plt.rc('text', usetex=True)
    for i, cov in enumerate(covfuncs):
        gp = GaussianProcess(cov, optimize=True, usegrads=False)
        gp.fit(X, y)
        xstar = np.arange(0, 2 * np.pi, step=0.01)
        Xstar = np.array([np.atleast_2d(u) for u in xstar])[:, 0]
        ymean, ystd = gp.predict(Xstar, return_std=True)

        lower, upper = ymean - 1.96 * np.sqrt(ystd), ymean + 1.96 * np.sqrt(ystd)
        plt.subplot(2, 2, i + 1)
        plt.plot(xstar, ymean, label='Posterior mean')
        plt.plot(xstar, np.sin(xstar), label='True function')
        plt.fill_between(xstar, lower, upper, alpha=0.4, label='95% confidence band', color=cm_bright[i])
        plt.grid()
        plt.title(titles[i])
    plt.legend(loc=0)
    plt.show()
