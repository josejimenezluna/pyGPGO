import numpy as np
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist


def l2norm_(X, Xstar):
    return cdist(X, Xstar)


def kronDelta(X, Xstar):
    n, m = X.shape[0], Xstar.shape[0]
    mat = np.zeros((n, m), dtype=np.int)
    for i in range(n):
        for j in range(m):
            if np.array_equal(X[i], Xstar[j]):
                mat[i, j] = 1
    return mat


default_bounds = {
    'l': [1e-4, 1],
    'sigmaf': [1e-4, 2],
    'sigman': [1e-6, 2],
    'v': [1e-3, 10],
    'gamma': [1e-3, 1.99],
    'alpha': [1e-3, 1e4]
}


class squaredExponential:
    def __init__(self, l=1, sigmaf=1.0, sigman=1e-6, bounds=None, parameters=['l', 'sigmaf',
                                                                              'sigman']):
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
        r = l2norm_(X, Xstar)
        return self.sigmaf * (np.exp(-.5 * r ** 2 / self.l ** 2)) + self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param='l'):
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
        r = l2norm_(X, Xstar)
        return self.sigmaf * (np.exp(-(r / self.l) ** self.gamma)) + \
               self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param):
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
    def __init__(self, alpha=1, l=1, sigmaf=1, sigman=1e-6, bounds=None, parameters = ['alpha',
                                                                                       'l',
                                                                                       'sigmaf',
                                                                                       'sigman']):
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
        r = l2norm_(X, Xstar)
        return self.sigmaf * ((1 + r ** 2 / (2 * self.alpha * self.l ** 2)) ** (-self.alpha))\
               + self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param):
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


class arcSin:
    def __init__(self, n, sigma=None):
        if sigma == None:
            self.sigma = np.eye(n)
        else:
            self.sigma = sigma

    def k(self, x, xstar):
        num = 2 * np.dot(np.dot(x[np.newaxis, :], self.sigma), xstar)
        a = 1 + 2 * np.dot(np.dot(x[np.newaxis, :], self.sigma), x)
        b = 1 + 2 * np.dot(np.dot(xstar[np.newaxis, :], self.sigma), xstar)
        res = num / np.sqrt(a * b)
        return (res)
