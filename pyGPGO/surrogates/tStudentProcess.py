import numpy as np
from scipy.linalg import inv

class tStudentProcess:
    def __init__(self, covfunc, nu=3.0):
        """
        t-Student Process regressor class. This class DOES NOT support Type II ML of covariance function
        hyperparameters.

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

        """
        self.covfunc = covfunc
        self.nu = nu

    def fit(self, X, y):
        """
        Fits a t-Student Process regressor

        Parameters
        ----------
        X: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the GP.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to X.

        """
        self.X = X
        self.y = y
        self.n1 = X.shape[0]

        self.K11 = self.covfunc.K(self.X, self.X)
        self.beta1 = np.dot(np.dot(self.y.T, inv(self.K11)), self.y)

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


if __name__ == '__main__':
    from pyGPGO.covfunc import *
    import matplotlib.pyplot as plt

    x = np.arange(0, 2 * np.pi + 0.01, step=np.pi / 3)
    y = np.sin(x)
    X = np.array([np.atleast_2d(u) for u in x])[:, 0]

    # Specify covariance function
    sexp = matern32()
    # Instantiate GPRegressor class
    gp = tStudentProcess(sexp)
    # Fit the model to the data
    gp.fit(X, y)

    # Predict on new data
    xstar = np.arange(0, 2 * np.pi, step=0.01)
    Xstar = np.array([np.atleast_2d(u) for u in xstar])[:, 0]
    ymean, ystd = gp.predict(Xstar, return_std=True)

    # Confidence interval bounds
    lower, upper = ymean - 1.96 * np.sqrt(ystd), ymean + 1.96 * np.sqrt(ystd)

    # Plot values
    plt.figure()
    plt.plot(xstar, ymean, label='Posterior mean')
    plt.plot(xstar, np.sin(xstar), label='True function')
    plt.fill_between(xstar, lower, upper, alpha=0.4, label=r'95\% confidence band')
    plt.grid()
    plt.legend(loc=0)
    plt.show()

