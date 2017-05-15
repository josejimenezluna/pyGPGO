import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

class RandomForest:
    def __init__(self, **params):
        """
        Wrapper around sklearn's Random Forest implementation for pyGPGO.
        Random Forests can also be used for surrogate models in Bayesian Optimization.
        An estimate of 'posterior' variance can be obtained by using the `impurity`
        criterion value in each subtree.

        Parameters
        ----------
        params: tuple, optional
            Any parameters to pass to `RandomForestRegressor`. Defaults to sklearn's.

        """
        self.params = params

    def fit(self, X, y):
        """
        Fit a Random Forest model to data `X` and targets `y`.

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
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, Xstar, return_std = True, eps = 1e-6):
        """
        Predicts 'posterior' mean and variance for the RF model.

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
        ymean = self.model.predict(Xstar)
        if return_std:
            std = np.zeros(len(Xstar))
            trees = self.model.estimators_

            for tree in trees:
                var_tree = tree.tree_.impurity[tree.apply(Xstar)]
                var_tree = np.clip(var_tree, eps, np.inf)
                mean_tree = tree.predict(Xstar)
                std += var_tree + mean_tree ** 2

            std /= len(trees)
            std -= ymean ** 2
            std = np.sqrt(np.clip(std, eps, np.inf))
            return ymean, std
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

class ExtraForest:
    def __init__(self, **params):
        """
        Wrapper around sklearn's ExtraTreesRegressor implementation for pyGPGO.
        Random Forests can also be used for surrogate models in Bayesian Optimization.
        An estimate of 'posterior' variance can be obtained by using the `impurity`
        criterion value in each subtree.

        Parameters
        ----------
        params: tuple, optional
            Any parameters to pass to `RandomForestRegressor`. Defaults to sklearn's.

        """
        self.params = params

    def fit(self, X, y):
        """
        Fit a Random Forest model to data `X` and targets `y`.

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
        self.model = ExtraTreesRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, Xstar, return_std = True, eps = 1e-6):
        """
        Predicts 'posterior' mean and variance for the RF model.

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
        ymean = self.model.predict(Xstar)
        if return_std:
            std = np.zeros(len(Xstar))
            trees = self.model.estimators_

            for tree in trees:
                var_tree = tree.tree_.impurity[tree.apply(Xstar)]
                var_tree = np.clip(var_tree, eps, np.inf)
                mean_tree = tree.predict(Xstar)
                std += var_tree + mean_tree ** 2

            std /= len(trees)
            std -= ymean ** 2
            std = np.sqrt(np.clip(std, eps, np.inf))
            return ymean, std
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