import numpy as np
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist

default_bounds = {
    'l': [1e-4, 1],
    'sigmaf': [1e-4, 2],
    'sigman': [1e-6, 2],
    'v': [1e-3, 10],
    'gamma': [1e-3, 1.99],
    'alpha': [1e-3, 1e4]
}


def l2norm_(X, Xstar):
    """
    Wrapper function to compute the L2 norm

    Parameters
    ----------
    X: np.ndarray, shape=((n, nfeatures))
        Instances.
    Xstar: np.ndarray, shape=((m, nfeatures))
        Instances

    Returns
    -------
    cdist: np.ndarray, shape=((n, m))
        Pairwise euclidian distance between row pairs of `X` and `Xstar`.
    """
    return cdist(X, Xstar)


def kronDelta(X, Xstar):
    """
    Computes Kronecker delta for rows in X and Xstar.

    Parameters
    ----------
    X: np.ndarray, shape=((n, nfeatures))
        Instances.
    Xstar: np.ndarray, shape((m, nfeatures))
        Instances.

    Returns
    -------
    kron: np.ndarray, shape=((n, m))
        Kronecker delta between row pairs of `X` and `Xstar`.
    """
    n, m = X.shape[0], Xstar.shape[0]
    mat = np.zeros((n, m), dtype=np.int)
    for i in range(n):
        for j in range(m):
            if np.array_equal(X[i], Xstar[j]):
                mat[i, j] = 1
    return mat


class squaredExponential:
    def __init__(self, l=1, sigmaf=1.0, sigman=1e-6, bounds=None, parameters=['l', 'sigmaf',
                                                                              'sigman']):
        """
        Squared exponential kernel class.

        Parameters
        ----------
        l: float
            Characteristic length-scale. Units in input space in which posterior GP values do not
            change significantly.
        sigmaf: float
            Signal variance. Controls the overall scale of the covariance function.
        sigman: float
            Noise variance. Additive noise in output space.
        bounds: list
            List of tuples specifying hyperparameter range in optimization procedure.
        parameters: list
            List of strings specifying which hyperparameters should be optimized.
        """
        self.l = l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def K(self, X, Xstar):
        """
        Computes covariance function values over `X` and `Xstar`.

        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances

        Returns
        -------
        cov: np.ndarray, shape=((n, m))
            Computed covariance matrix.
        """
        r = l2norm_(X, Xstar)
        return self.sigmaf * (np.exp(-.5 * r ** 2 / self.l ** 2)) + self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param='l'):
        """
        Computes gradient matrix for instances `X`, `Xstar` and hyperparameter `param`.

        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances
        param: str
            Parameter to compute gradient matrix for.

        Returns
        -------
        param_grad: np.ndarray, shape=((n, m))
            Gradient matrix for parameter `param`.
        """
        if param == 'l':
            r = l2norm_(X, Xstar)
            num = r ** 2 * np.exp(-r ** 2 / (2 * self.l ** 2))
            den = self.l ** 3
            l_grad = num / den
            return (l_grad)
        elif param == 'sigmaf':
            r = l2norm_(X, Xstar)
            sigmaf_grad = (np.exp(-.5 * r ** 2 / self.l ** 2))
            return (sigmaf_grad)

        elif param == 'sigman':
            sigman_grad = kronDelta(X, Xstar)
            return (sigman_grad)

        else:
            raise ValueError('Param not found')


class matern:
    def __init__(self, v=1, l=1, sigmaf=1, sigman=1e-6, bounds=None, parameters=['v',
                                                                                 'l',
                                                                                 'sigmaf',
                                                                                 'sigman']):
        """
        Matern kernel class.

        Parameters
        ----------
        v: float
            Scale-mixture hyperparameter of the Matern covariance function.
        l: float
            Characteristic length-scale. Units in input space in which posterior GP values do not
            change significantly.
        sigmaf: float
            Signal variance. Controls the overall scale of the covariance function.
        sigman: float
            Noise variance. Additive noise in output space.
        bounds: list
            List of tuples specifying hyperparameter range in optimization procedure.
        parameters: list
            List of strings specifying which hyperparameters should be optimized.
        """
        self.v, self.l = v, l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def K(self, X, Xstar):
        """
        Computes covariance function values over `X` and `Xstar`.

        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances

        Returns
        -------
        cov: np.ndarray, shape=((n, m))
            Computed covariance matrix.
        """
        r = l2norm_(X, Xstar)
        bessel = kv(self.v, np.sqrt(2 * self.v) * r / self.l)
        f = 2 ** (1 - self.v) / gamma(self.v) * (np.sqrt(2 * self.v) * r / self.l) ** self.v
        res = f * bessel
        res[np.isnan(res)] = 1
        res = self.sigmaf * res + self.sigman * kronDelta(X, Xstar)
        return (res)


class gammaExponential:
    def __init__(self, gamma=1, l=1, sigmaf=1, sigman=1e-6, bounds=None, parameters=['gamma',
                                                                                     'l',
                                                                                     'sigmaf',
                                                                                     'sigman']):
        """
        Gamma-exponential kernel class.

        Parameters
        ----------
        gamma: float
            Hyperparameter of the Gamma-exponential covariance function.
        l: float
            Characteristic length-scale. Units in input space in which posterior GP values do not
            change significantly.
        sigmaf: float
            Signal variance. Controls the overall scale of the covariance function.
        sigman: float
            Noise variance. Additive noise in output space.
        bounds: list
            List of tuples specifying hyperparameter range in optimization procedure.
        parameters: list
            List of strings specifying which hyperparameters should be optimized.
        """
        self.gamma = gamma
        self.l = l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def K(self, X, Xstar):
        """
        Computes covariance function values over `X` and `Xstar`.

        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances

        Returns
        -------
        cov: np.ndarray, shape=((n, m))
            Computed covariance matrix.
        """
        r = l2norm_(X, Xstar)
        return self.sigmaf * (np.exp(-(r / self.l) ** self.gamma)) + \
               self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param):
        """
        Computes gradient matrix for instances `X`, `Xstar` and hyperparameter `param`.

        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances
        param: str
            Parameter to compute gradient matrix for.

        Returns
        -------
        param_grad: np.ndarray, shape=((n, m))
            Gradient matrix for parameter `param`.
        """
        if param == 'gamma':
            eps = 10e-6
            r = l2norm_(X, Xstar) + eps
            first = -np.exp(- (r / self.l) ** self.gamma)
            sec = (r / self.l) ** self.gamma * np.log(r / self.l)
            gamma_grad = first * sec
            return (gamma_grad)
        elif param == 'l':
            r = l2norm_(X, Xstar)
            num = self.gamma * np.exp(-(r / self.l) ** self.gamma) * (r / self.l) ** self.gamma
            l_grad = num / self.l
            return (l_grad)
        elif param == 'sigmaf':
            r = l2norm_(X, Xstar)
            sigmaf_grad = (np.exp(-(r / self.l) ** self.gamma))
            return (sigmaf_grad)
        elif param == 'sigman':
            sigman_grad = kronDelta(X, Xstar)
            return (sigman_grad)
        else:
            raise ValueError('Param not found')


class rationalQuadratic:
    def __init__(self, alpha=1, l=1, sigmaf=1, sigman=1e-6, bounds=None, parameters=['alpha',
                                                                                     'l',
                                                                                     'sigmaf',
                                                                                     'sigman']):
        """
        Rational-quadratic kernel class.

        Parameters
        ----------
        alpha: float
            Hyperparameter of the rational-quadratic covariance function.
        l: float
            Characteristic length-scale. Units in input space in which posterior GP values do not
            change significantly.
        sigmaf: float
            Signal variance. Controls the overall scale of the covariance function.
        sigman: float
            Noise variance. Additive noise in output space.
        bounds: list
            List of tuples specifying hyperparameter range in optimization procedure.
        parameters: list
            List of strings specifying which hyperparameters should be optimized.
        """
        self.alpha = alpha
        self.l = l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def K(self, X, Xstar):
        """
        Computes covariance function values over `X` and `Xstar`.

        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances

        Returns
        -------
        cov: np.ndarray, shape=((n, m))
            Computed covariance matrix.
        """
        r = l2norm_(X, Xstar)
        return self.sigmaf * ((1 + r ** 2 / (2 * self.alpha * self.l ** 2)) ** (-self.alpha)) \
               + self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param):
        """
        Computes gradient matrix for instances `X`, `Xstar` and hyperparameter `param`.

        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances
        param: str
            Parameter to compute gradient matrix for.

        Returns
        -------
        param_grad: np.ndarray, shape=((n, m))
            Gradient matrix for parameter `param`.
        """
        if param == 'alpha':
            r = l2norm_(X, Xstar)
            one = (r ** 2 / (2 * self.alpha * self.l ** 2) + 1) ** (-self.alpha)
            two = r ** 2 / ((2 * self.alpha * self.l ** 2) * (r ** 2 / (2 * self.alpha * self.l ** 2) + 1))
            three = np.log(r ** 2 / (2 * self.alpha * self.l ** 2) + 1)
            alpha_grad = one * (two - three)
            return (alpha_grad)
        elif param == 'l':
            r = l2norm_(X, Xstar)
            num = r ** 2 * (r ** 2 / (2 * self.alpha * self.l ** 2) + 1) ** (-self.alpha - 1)
            l_grad = num / self.l ** 3
            return (l_grad)
        elif param == 'sigmaf':
            r = l2norm_(X, Xstar)
            sigmaf_grad = (1 + r ** 2 / (2 * self.alpha * self.l ** 2)) ** (-self.alpha)
            return (sigmaf_grad)
        elif param == 'sigman':
            sigman_grad = kronDelta(X, Xstar)
            return (sigman_grad)
        else:
            raise ValueError('Param not found')

# DEPRECATED
# class arcSin:
#     def __init__(self, n, sigma=None):
#         if sigma == None:
#             self.sigma = np.eye(n)
#         else:
#             self.sigma = sigma
#
#     def k(self, x, xstar):
#         num = 2 * np.dot(np.dot(x[np.newaxis, :], self.sigma), xstar)
#         a = 1 + 2 * np.dot(np.dot(x[np.newaxis, :], self.sigma), x)
#         b = 1 + 2 * np.dot(np.dot(xstar[np.newaxis, :], self.sigma), xstar)
#         res = num / np.sqrt(a * b)
#         return (res)