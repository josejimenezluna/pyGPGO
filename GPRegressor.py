import numpy as np
from scipy.linalg import cholesky, solve


class GPRegressor:
	def __init__(self, covfunc, sigma = 0):
		self.covfunc = covfunc
		self.sigma = sigma
	def fit(self, X, y):
		# Preallocate stuff
		self.X = X
		self.y = y
		self.nsamples = self.X.shape[0]
		# Compute similarity
		self.K = self.covfunc.K(self.X, self.X)
		# Compute common parameters
		self.L = cholesky(self.K + self.sigma * np.eye(self.nsamples)).T
		self.alpha = solve(self.L.T, solve(self.L, y))
		self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(np.diag(self.L)) - self.nsamples/2 * np.log(2 * np.pi)

	def predict(self, Xstar, return_std = False):
		Xstar = np.atleast_2d(Xstar)
		kstar = self.covfunc.K(self.X, Xstar).T
		fmean = np.dot(kstar, self.alpha)
		v = solve(self.L, kstar.T)
		fcov = self.covfunc.K(Xstar, Xstar) - np.dot(v.T, v)
		if return_std:
			fcov = np.diag(fcov)
		return fmean, fcov

	def update(self, xnew, ynew):
		y = np.concatenate((self.y, ynew), axis = 0)
		X = np.concatenate((self.X, xnew), axis = 0)
		self.fit(X, y)

