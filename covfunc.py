"""
Example of covariance functions for Gaussian processes
"""
import numpy as np
from scipy.special import gamma, kv

def l2norm(x, xstar):
	return(np.sqrt(np.sum((x - xstar)**2, axis = 1)))

class squaredExponential:
	def __init__(self, l = 1):
		self.l = l
	def k(self, x, xstar):
		r = l2norm(x, xstar)
		return(np.exp(-.5 * r ** 2) / self.l ** 2)

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

class gammaExponential:
	def __init__(self, gamma = 1, l = 1):
		self.gamma = gamma
		self.l = l
	def k(self, x, xstar):
		r = l2norm(x, xstar)
		return(np.exp(-(r / self.l) ** self.gamma))

class rationalQuadratic:
	def __init__(self, alpha = 1, l = 1):
		self.alpha = alpha
		self.l = l
	def k(self, x, xstar):
		r = l2norm(x, xstar)
		return((1 + r**2/(2 * self.alpha * self.l **2))**(-self.alpha))

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
