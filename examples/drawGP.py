#######################################
# pyGPGO examples
# drawGP: Samples from a GP prior.
#######################################

import numpy as np
from numpy.random import multivariate_normal
from pyGPGO.covfunc import squaredExponential
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(93)
    # Equally spaced values of Xstar
    Xstar = np.arange(0, 2 * np.pi, step=np.pi / 24)
    Xstar = np.array([np.atleast_2d(x) for x in Xstar])[:, 0]
    sexp = squaredExponential()
    # By default assume mean 0
    m = np.zeros(Xstar.shape[0])
    # Compute squared-exponential matrix
    K = sexp.K(Xstar, Xstar)

    n_samples = 3
    # Draw samples from multivariate normal
    samples = multivariate_normal(m, K, size=n_samples)

    # Plot values
    x = Xstar.flatten()
    plt.figure()
    for i in range(n_samples):
        plt.plot(x, samples[i], label='GP sample {}'.format(i + 1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sampled GP priors from Squared Exponential kernel')
    plt.grid()
    plt.legend(loc=0)
    plt.show()
