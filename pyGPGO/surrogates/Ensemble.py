import numpy as np

class Ensemble:
    def __init__(self, members):
        """
        Ensemble of surrogates class. Based on experimental idea of D.W. Shimer


        Parameters
        ----------
        members: list of member surrogate functions for the ensemble

        Attributes
        ----------
        members: list of objects
            List of surrogate function objects.
        By D. W. Shimer
        """
        self.members = members
    
    def fit(self, X, y):
        """
        Fits an emsemble surrogate function

        Parameters
        ----------
        X: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the ensemble.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to X.

        """
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        for member in self.members:
            member.fit(X,y)

    def update(self, xnew, ynew):
        """
        Updates the ensemble model with observations `xnew` and targets `ynew`.
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
    
    def predict(self, Xstar, return_std = True, eps = 1e-6):
        """
        Predicts 'posterior' mean and variance for the ensemble model.
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
        mean_array = []
        cov_array = []
        for member in self.members:
            mean, std = member.predict(Xstar, return_std=True)
            mean_array.append(mean)
            cov_array.append(std)
        out_mean = np.mean(mean_array, axis=0)
        out_std = np.mean(cov_array, axis=0)
        return out_mean, out_std