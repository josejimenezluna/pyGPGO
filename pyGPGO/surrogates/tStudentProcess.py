import numpy as np
from collections import OrderedDict
from numpy.linalg import slogdet
from scipy.linalg import inv
from scipy.optimize import minimize
from scipy.special import gamma


def logpdf(x, df, mu, Sigma):
    """
    Marginal log-likelihood of a Student-t Process

    Parameters
    ----------
    x: array-like
        Point to be evaluated
    df: float
        Degrees of freedom (>2.0)
    mu: array-like
        Mean of the process.
    Sigma: array-like
        Covariance matrix of the process.

    Returns
    -------
    logp: float
        log-likelihood 

    """
    d = len(x)
    x = np.atleast_2d(x)
    xm = x - mu
    V = df * Sigma
    V_inv = np.linalg.inv(V)
    _, logdet = slogdet(np.pi * V)

    logz = -gamma(df / 2.0 + d / 2.0) + gamma(df / 2.0) + 0.5 * logdet
    logp = -0.5 * (df + d) * np.log(1 + np.sum(np.dot(xm, V_inv) * xm, axis=1))

    logp = logp - logz

    return logp[0]


class tStudentProcess:
    def __init__(self, covfunc, nu=3.0, optimize=False):
        """
        t-Student Process regressor class.
        This class DOES NOT support gradients in ML estimation yet.

        Parameters
        ----------
        covfunc: instance from a class of covfunc module
            An instance from a class from the `covfunc` module.
        nu: float
            (>2.0) Degrees of freedom

        Attributes
        ----------
        covfunc: object
            Internal covariance function.
        nu: float
            Degrees of freedom.
        optimize: bool
            Whether to optimize covariance function hyperparameters.

        """
        self.covfunc = covfunc
        self.nu = nu
        self.optimize = optimize

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
        self.covfunc = self.covfunc.__class__(**k_param, bounds=self.covfunc.bounds)

        # This fixes recursion
        original_opt = self.optimize
        self.optimize = False
        self.fit(self.X, self.y)
        self.optimize = original_opt

        return (- self.logp)

    def optHyp(self, param_key, param_bounds, n_trials=5):
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

            res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=param_bounds)
            xs.append(res.x)
            fs.append(res.fun)

        argmin = np.argmin(fs)
        opt_param = xs[argmin]
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.covfunc = self.covfunc.__class__(**k_param, bounds=self.covfunc.bounds)

    def fit(self, X, y):
        """
        Fits a t-Student Process regressor

        Parameters
        ----------
        X: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the GP.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to `X`.

        """
        self.X = X
        self.y = y
        self.n1 = X.shape[0]

        if self.optimize:
            self.optHyp(param_key=self.covfunc.parameters, param_bounds=self.covfunc.bounds)

        self.K11 = self.covfunc.K(self.X, self.X)
        self.beta1 = np.dot(np.dot(self.y.T, inv(self.K11)), self.y)
        self.logp = logpdf(self.y, self.nu, mu=np.zeros(self.n1), Sigma=self.K11)

    def predict(self, Xstar, return_std=False):
        """
        Returns mean and covariances for the posterior t-Student process.

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
        self.K21 = self.covfunc.K(Xstar, self.X)
        self.K22 = self.covfunc.K(Xstar, Xstar)
        self.K12 = self.covfunc.K(self.X, Xstar)
        self.K22_tilde = self.K22 - np.dot(np.dot(self.K21, inv(self.K11)), self.K12)

        phi2 = np.dot(np.dot(self.K21, inv(self.K11)), self.y)
        cov = (self.nu + self.beta1 - 2) / (self.nu + self.n1 - 2) * self.K22_tilde
        if return_std:
            return phi2, np.diag(cov)
        return phi2, cov

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
