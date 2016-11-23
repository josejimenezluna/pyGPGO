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
	def ExpectedImprovement(self, xbest, mean, var):
		return norm.cdf((mean - xbest) / var)
	def ProbabilityImprovement(self, xbest, mean, var):
		z = (mean - xbest) / var
		return (mean - xbest) * norm.cdf(z) + var * norm.pdf(z)
	def UCB(self, xbest, mean, var, beta):
		return mean + beta * var
	def eval(self, xbest, mean, var):
		return self.f(self, xbest, mean, var, **self.params)
	
