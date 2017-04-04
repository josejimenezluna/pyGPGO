# work in progress
import numpy as np
import scipy as sp
import theano
import theano.tensor as tt
import theano.tensor.nlinalg
import pymc3 as pm
from pyGPGO.covfunc import squaredExponential, matern
from pyGPGO.surrogates.GaussianProcess import GaussianProcess

covariance_equivalence = {'squaredExponential': pm.gp.cov.ExpQuad,
                          'matern': pm.gp.cov.Matern52}

class GaussianProcessMCMC:
    def __init__(self, covfunc):
        self.covfunc = covfunc

    def _extractParam(self, unittrace, covparams):
        d = {}
        for key, value in unittrace.items():
            if key in covparams:
                d[key] = value
        if 'v' in covparams:
            d['v'] = 5/2
        return d

    def fit(self, X, y, niter=2000, burnin=1000):
        """
        Fits a Gaussian Process regressor using MCMC.

        Parameters
        ----------
        X: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the GP.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to X.
        niter: int
            Number of iterations to run MCMC.
        burnin: int
            Burn-in iterations to discard at the beginnint
        """
        self.X = X
        self.y = y
        self.niter = niter
        self.burnin = burnin

        with pm.Model() as model:
            l = pm.Uniform('l', 0, 10)

            log_s2_f = pm.Uniform('log_s2_f', lower=-7, upper=5)
            s2_f = pm.Deterministic('sigmaf', tt.exp(log_s2_f))

            log_s2_n = pm.Uniform('log_s2_n', lower=-7, upper=5)
            s2_n = pm.Deterministic('sigman', tt.exp(log_s2_n))

            f_cov = s2_f * covariance_equivalence[type(self.covfunc).__name__](1, l)

            y_obs = pm.gp.GP('y_obs', cov_func=f_cov, sigma=s2_n, observed={'X': self.X, 'Y': self.y})
        with model:
            self.trace = list(pm.sample(niter)[burnin:])

    def predict(self, Xstar, return_std=False, nsamples=100):
        """
        Returns mean and covariances for each posterior sampled Gaussian Process.

        Parameters
        ----------
        Xstar: np.ndarray, shape=((nsamples, nfeatures))
            Testing instances to predict.
        return_std: bool
            Whether to return the standard deviation of the posterior process. Otherwise,
            it returns the whole covariance matrix of the posterior process.
        nsamples:
            Number of posterior MCMC samples to consider.

        Returns
        -------
        np.ndarray
            Mean of the posterior process for each MCMC sample and Xstar.
        np.ndarray
            Covariance posterior process for each MCMC sample and Xstar.
        """
        chunk = self.trace[::-1][:nsamples]
        post_mean = []
        post_var = []
        for posterior_sample in chunk:
            params = self._extractParam(posterior_sample, self.covfunc.parameters)
            covfunc = self.covfunc.__class__(**params)
            gp = GaussianProcess(covfunc)
            gp.fit(self.X, self.y)
            m, s = gp.predict(Xstar, return_std=return_std)
            post_mean.append(m)
            post_var.append(s)
        return np.array(post_mean), np.array(post_var)

    def update(self, xnew, ynew):
        """
        Updates the internal model with `xnew` and `ynew` instances.

        Parameters
        ----------
        xnew: np.ndarray, shape=((m, nfeatures))
            New training instances to update the model with.
        ynew: np.ndarray, shape=((m,))
            New training targets to update the model with.
        """
        y = np.concatenate((self.y, ynew), axis=0)
        X = np.concatenate((self.X, xnew), axis=0)
        self.fit(X, y, self.niter, self.burnin)


if __name__ == '__main__':
    # np.random.seed(20090425)
    # n = 20
    # X = np.sort(3 * np.random.rand(n))[:, None]
    # with pm.Model() as model:
    #     # f(x)
    #     l_true = 0.3
    #     s2_f_true = 1.0
    #     cov = s2_f_true * pm.gp.cov.ExpQuad(1, l_true)
    #
    #     # noise, epsilon
    #     s2_n_true = 0.1
    #     K_noise = s2_n_true ** 2 * tt.eye(n)
    #     K = cov(X) + K_noise
    #
    # K = theano.function([], cov(X) + K_noise)()
    # y = np.random.multivariate_normal(np.zeros(n), K)
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 6, 60)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.2, X.shape[0])

    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp)
    gp.fit(X, y)
    Z = np.linspace(0, 6, 100)[:, None]
    post_mean, post_var = gp.predict(Z, return_std=True, nsamples=200)

    import matplotlib.pyplot as plt
    for i in range(100):
        plt.plot(Z.flatten(), post_mean[i])
    plt.show()
