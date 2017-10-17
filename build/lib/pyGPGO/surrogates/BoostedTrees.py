import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class BoostedTrees:
    def __init__(self, q1=.16, q2=.84,**params):
        """
        Gradient boosted trees as surrogate model for Bayesian Optimization.
        Uses quantile regression for an estimate of the 'posterior' variance.
        In practice, the std is computed as (`q2` - `q1`) / 2.
        Relies on `sklearn.ensemble.GradientBoostingRegressor`

        Parameters
        ----------
        q1: float
            First quantile.
        q2: float
            Second quantile
        params: tuple
            Extra parameters to pass to `GradientBoostingRegressor`

        """
        self.params = params
        self.q1 = q1
        self.q2 = q2
        self.eps = 1e-1

    def fit(self, X, y):
        """
        Fit a GBM model to data `X` and targets `y`.

        Parameters
        ----------
        X : array-like
            Input values.
        y: array-like
            Target values.
        """
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.modq1 = GradientBoostingRegressor(loss='quantile', alpha=self.q1, **self.params)
        self.modq2 = GradientBoostingRegressor(loss='quantile', alpha=self.q2, **self.params)
        self.mod = GradientBoostingRegressor(loss = 'ls', **self.params)
        self.modq1.fit(self.X, self.y)
        self.modq2.fit(self.X, self.y)
        self.mod.fit(self.X, self.y)

    def predict(self, Xstar, return_std = True):
        """
        Predicts 'posterior' mean and variance for the GBM model.

        Parameters
        ----------
        Xstar: array-like
            Input values.
        return_std: bool, optional
            Whether to return posterior variance estimates. Default is `True`.
        eps: float, optional
            Floating precision value for negative variance estimates. Default is `1e-6`


        Returns
        -------
        array-like:
            Posterior predicted mean.
        array-like:
            Posterior predicted std

        """
        Xstar = np.atleast_2d(Xstar)
        ymean = self.mod.predict(Xstar)
        if return_std:
            q1pred = self.modq1.predict(Xstar)
            q2pred = self.modq2.predict(Xstar)
            ystd = (q2pred - q1pred) / 2 + self.eps
            return ymean, ystd
        return ymean

    def update(self, xnew, ynew):
        """
        Updates the internal RF model with observations `xnew` and targets `ynew`.

        Parameters
        ----------
        xnew: array-like
            New observations.
        ynew: array-like
            New targets.
        """
        y = np.concatenate((self.y, ynew), axis=0)
        X = np.concatenate((self.X, xnew), axis=0)
        self.fit(X, y)