import numpy as np
from scipy.stats import norm

class Acquisition:
	def __init__(self, mode, **params):
		self.params = params
		if mode == 'ExpectedImprovement':
			self.f = self.ExpectedImprovement
		if mode == 'ProbabilityImprovement':
			self.f = self.ProbabilityImprovement
		if mode == 'UCB':
			self.f = self.YetAnotherWay
		else:
			raise ValueError('Not recognised acquisition function')
	def ExpectedImprovement(self, xbest, mean, std):
		return norm.cdf((mean - xbest) / std)
	def ProbabilityImprovement(self, xbest, mean, std):
		z = (mean - xbest) / std
		return (mean - xbest) * norm.cdf(z) + var * norm.pdf(z)
	def UCB(self, xbest, mean, std, beta):
		return mean + beta * std
	def eval(self, xbest, mean, var):
		return self.f(self, xbest, mean, var, **self.params)
	
