import numpy as np
from scipy.linalg import cholesky, solve
from collections import OrderedDict
from scipy.optimize import minimize

class GaussianProcess:
    def __init__(self, covfunc, optimize=False, usegrads=False, mprior=0):
        """
        Gaussian Process regressor class. Based on Rasmussen & Williams [1]_ algorithm 2.1.

        Parameters
        ----------
        covfunc: instance from a class of covfunc module
            Covariance function. An instance from a class in the `covfunc` module.
        optimize: bool:
            Whether to perform covariance function hyperparameter optimization.
        usegrads: bool
            Whether to use gradient information on hyperparameter optimization. Only used
            if `optimize=True`.

        Attributes
        ----------
        covfunc: object
            Internal covariance function.
        optimize: bool
            User chosen optimization configuration.
        usegrads: bool
            Gradient behavior
        mprior: float
            Explicit value for the mean function of the prior Gaussian Process.

        Notes
        -----
        [1] Rasmussen, C. E., & Williams, C. K. I. (2004). Gaussian processes for machine learning.
        International journal of neural systems (Vol. 14). http://doi.org/10.1142/S0129065704001899
        """
        self.covfunc = covfunc
        self.optimize = optimize
        self.usegrads = usegrads
        self.mprior = mprior

    def getcovparams(self):
        """
        Returns current covariance function hyperparameters

        Returns
        -------
        dict
            Dictionary containing covariance function hyperparameters
        """
        d = {}
        for param in self.covfunc.parameters:
            d[param] = self.covfunc.__dict__[param]
        return d

    def fit(self, X, y):
        """
        Fits a Gaussian Process regressor

        Parameters
        ----------
        X: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the GP.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to X.

        """
        self.X = X
        self.y = y
        self.nsamples = self.X.shape[0]
        if self.optimize:
            grads = None
            if self.usegrads:
                grads = self._grad
            self.optHyp(param_key=self.covfunc.parameters, param_bounds=self.covfunc.bounds, grads=grads)

        self.K = self.covfunc.K(self.X, self.X)
        self.L = cholesky(self.K).T
        self.alpha = solve(self.L.T, solve(self.L, y - self.mprior))
        self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(np.log(np.diag(self.L))) - self.nsamples / 2 * np.log(
            2 * np.pi)

    def param_grad(self, k_param):
        """
        Returns gradient over hyperparameters. It is recommended to use `self._grad` instead.

        Parameters
        ----------
        k_param: dict
            Dictionary with keys being hyperparameters and values their queried values.

        Returns
        -------
        np.ndarray
            Gradient corresponding to each hyperparameters. Order given by `k_param.keys()`
        """
        k_param_key = list(k_param.keys())
        covfunc = self.covfunc.__class__(**k_param)
        n = self.X.shape[0]
        K = covfunc.K(self.X, self.X)
        L = cholesky(K).T
        alpha = solve(L.T, solve(L, self.y))
        inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(K)
        grads = []
        for param in k_param_key:
            gradK = covfunc.gradK(self.X, self.X, param=param)
            gradK = .5 * np.trace(np.dot(inner, gradK))
            grads.append(gradK)
        return np.array(grads)

    def _lmlik(self, param_vector, param_key):
        """
        Returns marginal negative log-likelihood for given covariance hyperparameters.

        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.

        Returns
        -------
        float
            Negative log-marginal likelihood for chosen hyperparameters.

        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        self.covfunc = self.covfunc.__class__(**k_param)

        # This fixes recursion
        original_opt = self.optimize
        original_grad = self.usegrads
        self.optimize = False
        self.usegrads = False

        self.fit(self.X, self.y)

        self.optimize = original_opt
        self.usegrads = original_grad
        return (- self.logp)

    def _grad(self, param_vector, param_key):
        """
        Returns gradient for each hyperparameter, evaluated at a given point.

        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.

        Returns
        -------
        np.ndarray
            Gradient for each evaluated hyperparameter.

        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        return - self.param_grad(k_param)

    def optHyp(self, param_key, param_bounds, grads=None, n_trials=5):
        """
        Optimizes the negative marginal log-likelihood for given hyperparameters and bounds.
        This is an empirical Bayes approach (or Type II maximum-likelihood).

        Parameters
        ----------
        param_key: list
            List of hyperparameters to optimize.
        param_bounds: list
            List containing tuples defining bounds for each hyperparameter to optimize over.

        """
        xs = [[1, 1, 1]]
        fs = [self._lmlik(xs[0], param_key)]
        for trial in range(n_trials):
            x0 = []
            for param, bound in zip(param_key, param_bounds):
                x0.append(np.random.uniform(bound[0], bound[1], 1)[0])
            if grads is None:
                res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=param_bounds)
            else:
                res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=param_bounds, jac=grads)
            xs.append(res.x)
            fs.append(res.fun)

        argmin = np.argmin(fs)
        opt_param = xs[argmin]
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.covfunc = self.covfunc.__class__(**k_param)

    def predict(self, Xstar, return_std=False):
        """
        Returns mean and covariances for the posterior Gaussian Process.

        Parameters
        ----------
        Xstar: np.ndarray, shape=((nsamples, nfeatures))
            Testing instances to predict.
        return_std: bool
            Whether to return the standard deviation of the posterior process. Otherwise,
            it returns the whole covariance matrix of the posterior process.

        Returns
        -------
        np.ndarray
            Mean of the posterior process for testing instances.
        np.ndarray
            Covariance of the posterior process for testing instances.
        """
        Xstar = np.atleast_2d(Xstar)
        kstar = self.covfunc.K(self.X, Xstar).T
        fmean = self.mprior + np.dot(kstar, self.alpha)
        v = solve(self.L, kstar.T)
        fcov = self.covfunc.K(Xstar, Xstar) - np.dot(v.T, v)
        if return_std:
            fcov = np.diag(fcov)
        return fmean, fcov

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
        self.fit(X, y)