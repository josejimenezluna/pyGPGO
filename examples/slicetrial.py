import matplotlib.pyplot as plt
import matplotlib.cm as cmap
cm = cmap.inferno

import numpy as np
import scipy as sp
import theano
import theano.tensor as tt
import theano.tensor.nlinalg
import sys
sys.path.insert(0, "../../..")
import pymc3 as pm


if __name__ == '__main__':
    np.random.seed(20090425)
    n = 20
    X = np.sort(3 * np.random.rand(n))[:, None]

    with pm.Model() as model:
        # f(x)
        l_true = 0.3
        s2_f_true = 1.0
        cov = s2_f_true * pm.gp.cov.ExpQuad(1, l_true)

        # noise, epsilon
        s2_n_true = 0.1
        K_noise = s2_n_true ** 2 * tt.eye(n)
        K = cov(X) + K_noise

    # evaluate the covariance with the given hyperparameters
    K = theano.function([], cov(X) + K_noise)()

    # generate fake data from GP with white noise (with variance sigma2)
    y = np.random.multivariate_normal(np.zeros(n), K)

    # Proper fitting
    Z = np.linspace(0, 3, 100)[:, None]

    with pm.Model() as model:
        # priors on the covariance function hyperparameters
        l = pm.Uniform('l', 0, 10)

        # uninformative prior on the function variance
        log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=5)
        s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

        # uninformative prior on the noise variance
        log_s2_n = pm.Uniform('log_s2_n', lower=-10, upper=5)
        s2_n = pm.Deterministic('s2_n', tt.exp(log_s2_n))

        # covariance functions for the function f and the noise
        f_cov = s2_f * pm.gp.cov.ExpQuad(1, l)

        y_obs = pm.gp.GP('y_obs', cov_func=f_cov, sigma=s2_n, observed={'X': X, 'Y': y})

    with model:
        trace = pm.sample(2000)

    pm.traceplot(trace[1000:], varnames=['l', 's2_f', 's2_n'],
                 lines={"l": l_true,
                        "s2_f": s2_f_true,
                        "s2_n": s2_n_true})

    # Predictive distribution
    with model:
        gp_samples = pm.gp.sample_gp(trace[1500:], y_obs, Z, random_seed=42)