"""
Example of covariance functions for Gaussian processes
"""
import numpy as np
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist

def l2norm(x, xstar):
	return(np.sqrt(np.sum((x - xstar)**2, axis = 1)))

def l2norm_(X, Xstar):
	return cdist(X, Xstar)

class squaredExponential:
	def __init__(self, l = 1, sigma2f = 1, sigma2n = 0):
		self.l = l
		self.sigma2f = sigma2f
		self.sigma2n = sigma2n

	def K(self, X, Xstar):
		r = l2norm_(X, Xstar)
		return(np.exp(-.5 * r ** 2 / self.l ** 2))

	def gradK(self, X, Xstar, param = 'l'):
		if param == 'l':
			r = l2norm_(X, Xstar)
			num = r**2 * np.exp(-r**2 /(2* self.l**2))
			den = self.l ** 3
			l_grad = num / den
			return(l_grad)
		else:
			raise ValueError('Param not found')

class matern:
	def __init__(self, v = 1, l = 1):
		self.v, self.l = v, l
	def k(self, x, xstar):
		r = l2norm(x, xstar)
		bessel = kv(self.v, np.sqrt(2 * self.v) * r / self.l)
		f = 2 ** (1 - self.v) / gamma(self.v) * (np.sqrt(2 * self.v) * r / self.l) ** self.v
		res = f * bessel 
		res[np.isnan(res)] = 1
		return(res)
	def K(self, X, Xstar):
		r = l2norm_(x, xstar)
		bessel = kv(self.v, np.sqrt(2 * self.v) * r / self.f)
		f = 2 ** (1 - self.v) / gamma(self.v) * (np.sqrt(2 * self.v) * r / self.l) ** self.v
		res = f * bessel
		res[np.isnan(res)] = 1
		return(res)

class gammaExponential:
	def __init__(self, gamma = 1, l = 1):
		self.gamma = gamma
		self.l = l

	def K(self, X, Xstar):
		r = l2norm_(X, Xstar)
		return(np.exp(-(r / self.l) ** self.gamma))

	def gradK(self, X, Xstar, param):
		if param == 'gamma':
			eps = 10e-6
			r = l2norm_(X, Xstar) + eps
			first = -np.exp(- (r / self.l) ** self.gamma)
			sec = (r / self.l) ** self.gamma * np.log(r / self.l)
			gamma_grad = first * sec
			return(gamma_grad)
		if param == 'l':
			r = l2norm_(X, Xstar)
			num = self.gamma * np.exp(-(r / self.l)**self.gamma) * (r / self.l) ** self.gamma
			l_grad = num / self.l
			return(l_grad)

class rationalQuadratic:
	def __init__(self, alpha = 1, l = 1):
		self.alpha = alpha
		self.l = l

	def K(self, X, Xstar):
		r = l2norm_(X, Xstar)
		return((1 + r**2/(2 * self.alpha * self.l **2))**(-self.alpha))

	def gradK(self, X, Xstar, param):
		if param == 'alpha':
			r = l2norm_(X, Xstar)
			one = (r**2/(2 * self.alpha * self.l **2) + 1)**(-self.alpha)
			two = r**2 / ((2 * self.alpha * self.l **2) * (r**2 / (2 * self.alpha * self.l**2) + 1))
			three = np.log(r**2/(2* self.alpha * self.l **2) + 1)
			alpha_grad = one * (two - three)
			return(alpha_grad)
		if param == 'l':
			r = l2norm_(X, Xstar)
			num = r ** 2 * (r ** 2 /(2 * self.alpha * self.l**2) + 1) ** (-self.alpha - 1)
			l_grad = num / self.l ** 3
			return(l_grad)
class arcSin:
	def __init__(self, n, sigma = None):
		if sigma == None:
			self.sigma = np.eye(n)
		else:
			self.sigma = sigma
	def k(self, x, xstar):
		num = 2 * np.dot(np.dot(x[np.newaxis, :], self.sigma), xstar)
		a = 1 + 2 * np.dot(np.dot(x[np.newaxis, :], self.sigma), x)
		b = 1 + 2 * np.dot(np.dot(xstar[np.newaxis, :], self.sigma), xstar)
		res = num / np.sqrt(a * b)
		return(res)		
