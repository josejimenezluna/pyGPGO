#######################################
# pyGPGO examples
# hyperopt: shows the gradient w.r.t. the characteristic length scale
# on a simple example.
#######################################


import numpy as np
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.covfunc import squaredExponential
import matplotlib.pyplot as plt


def gradient(gp, sexp):
    alpha = gp.alpha
    K = gp.K
    gradK = sexp.gradK(gp.X, gp.X, 'l')
    inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(K)
    return (.5 * np.trace(np.dot(inner, gradK)))


if __name__ == '__main__':
    x = np.arange(0, 2 * np.pi + 0.01, step=np.pi / 2)
    X = np.array([np.atleast_2d(u) for u in x])[:, 0]
    y = np.sin(x)

    logp = []
    grad = []
    length_scales = np.linspace(0.1, 2, 1000)

    for l in length_scales:
        sexp = squaredExponential(l=l)
        gp = GaussianProcess(sexp)
        gp.fit(X, y)
        logp.append(gp.logp)
        grad.append(gradient(gp, sexp))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(length_scales, logp)
    plt.title('Marginal log-likelihood')
    plt.xlabel('Characteristic length-scale l')
    plt.ylabel('log-likelihood')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(length_scales, grad, '--', color='red')
    plt.title('Gradient w.r.t. l')
    plt.xlabel('Characteristic length-scale l')
    plt.grid()
    plt.show()
