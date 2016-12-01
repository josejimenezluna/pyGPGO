import numpy as np
from GPRegressor import GPRegressor
from covfunc import *
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def gradient(gp, sexp):
	alpha = gp.alpha
	K = gp.K
	gradK = sexp.gradK(gp.X, gp.X)
	inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(K)
	return(.5 * np.trace(np.dot(inner, gradK))) 


if __name__ == '__main__':
	X = np.array([[1], [2], [7], [18]])
	y = np.array([0.5, 1.2, 0.6, 2])

	logp = []
	logsk = []
	grad = []
	gradsk = []
	l_range = np.linspace(0.01, 2, 100)

	for l in l_range:
		sexp = squaredExponential(l = l)
		gp = GPRegressor(sexp)
		gp.fit(X, y)
		logp.append(gp.logp)
		grad.append(gradient(gp, sexp))

		rbf = RBF(l, length_scale_bounds = 'fixed' )
		u = GaussianProcessRegressor(rbf)
		u.fit(X, y)
		logsk.append(u.log_marginal_likelihood_value_)
		#gradsk.append(u.log_marginal_likelihood(theta = np.array(), eval_gradient = True)[1])

	plt.plot(l_range, logp)
	plt.plot(l_range, logsk)
	plt.figure()
	plt.plot(l_range, grad)
	#plt.plot(l_range, gradsk)
	plt.show()
