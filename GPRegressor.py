import numpy as np
from scipy.linalg import cholesky, solve


class GPRegressor:
	def __init__(self, covfunc, sigma = 0):
		self.covfunc = covfunc
		self.sigma = sigma
	def _sim(self):
		self.K = np.empty((self.nsamples, self.nsamples))
		for i in range(self.nsamples):
			self.K[i, :] = self.covfunc.k(self.X, self.X[i])
	def fit(self, X, y):
		# Preallocate stuff
		self.X = X
		self.y = y
		self.nsamples = self.X.shape[0]
		# Compute similarity
		self._sim()
		# Compute common parameters
		self.L = cholesky(self.K + self.sigma * np.eye(self.nsamples)).T
		self.alpha = solve(self.L.T, solve(self.L, y))
		self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(np.diag(self.L)) - self.nsamples/2 * np.log(2 * np.pi)
	def predict(self, xstar):
		xstar = np.atleast_2d(xstar)
		# Compute similarity of xstar to X
		kstar = self.covfunc.k(self.X, xstar)
		# Compute prediction
		fmean = np.dot(kstar, self.alpha)
		v = solve(self.L, kstar)
		fvar = self.covfunc.k(xstar, xstar) - np.sum(v**2)
		return fmean, fvar
	def update(self, xnew, ynew):
		y = np.concatenate((self.y, ynew), axis = 0)
		X = np.concatenate((self.X, xnew), axis = 0)
		self.fit(X, y)

	
# Gaussian process implementation for regression
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from covfunc import *
	
	X = np.array([[-4], [-2.5], [0], [2]])
	y = np.array([-2, 0, 1, -1])
	sexp = squaredExponential(l = 1)

	gp = GPRegressor(sexp)
	gp.fit(X, y)
	x_test = np.linspace(-5, 5, num = 1000)
	yhat = np.array([gp.predict(x)[0] for x in x_test])
	yvar = np.array([gp.predict(x)[1] for x in x_test]).flatten()
	l, u = yhat - 1.96 * yvar, yhat + 1.96 * yvar
	plt.plot(X.flatten(), y, 'x', color = 'black', ms = 10, mew = 2)
	plt.fill_between(x_test, l, u, alpha = 0.5, edgecolor = 'blue')
	plt.show()
