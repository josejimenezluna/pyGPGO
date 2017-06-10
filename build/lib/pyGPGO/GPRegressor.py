import numpy as np
from scipy.linalg import cholesky, solve
from collections import OrderedDict
from scipy.optimize import minimize


class GPRegressor:
    def __init__(self, covfunc, sigma=0, optimize = False, usegrads = False):
        self.covfunc = covfunc
        self.sigma = sigma
        self.optimize = optimize
        self.usegrads = usegrads

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.nsamples = self.X.shape[0]
        if self.optimize:
            grads = None
            if self.usegrads:
                grads = self._grad
            self.optHyp(param_key=self.covfunc.parameters, param_bounds=self.covfunc.bounds, grads=grads)

        self.K = self.covfunc.K(self.X, self.X)
        self.L = cholesky(self.K + self.sigma * np.eye(self.nsamples)).T
        self.alpha = solve(self.L.T, solve(self.L, y))
        self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(np.log(np.diag(self.L))) - self.nsamples / 2 * np.log(
            2 * np.pi)

    def param_grad(self, k_param):
        k_param_key = list(k_param.keys())
        covfunc = self.covfunc.__class__(**k_param)
        n = self.X.shape[0]
        K = covfunc.K(self.X, self.X)
        L = cholesky(K + self.sigma * np.eye(n)).T
        alpha = solve(L.T, solve(L, self.y))
        grads = []
        for param in k_param_key:
            gradK = covfunc.gradK(self.X, self.X, param=param)
            inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(K)
            gradK = .5 * np.trace(np.dot(inner, gradK))
            grads.append(gradK)
        return np.array(grads)

    def _lmlik(self, param_vector, param_key):
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
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        return - self.param_grad(k_param)

    def optHyp(self, param_key, param_bounds, grads = None):
        x0 = np.repeat(1, len(param_key))
        if grads is None:
            res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=[param_bounds])
        else:
            res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=[param_bounds], jac=grads)
        opt_param = res.x
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.covfunc = self.covfunc.__class__(**k_param)

    def predict(self, Xstar, return_std=False):
        Xstar = np.atleast_2d(Xstar)
        kstar = self.covfunc.K(self.X, Xstar).T
        fmean = np.dot(kstar, self.alpha)
        v = solve(self.L, kstar.T)
        fcov = self.covfunc.K(Xstar, Xstar) - np.dot(v.T, v)
        if return_std:
            fcov = np.diag(fcov)
        return fmean, fcov

    def update(self, xnew, ynew):
        y = np.concatenate((self.y, ynew), axis=0)
        X = np.concatenate((self.X, xnew), axis=0)
        self.fit(X, y)
