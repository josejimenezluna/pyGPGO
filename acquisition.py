import numpy as np
from scipy.stats import norm

class Acquisition:
	def __init__(self, mode, **params):
		self.params = params
		if mode == 'ExpectedImprovement':
			self.f = self.ExpectedImprovement
		elif mode == 'ProbabilityImprovement':
			self.f = self.ProbabilityImprovement
		elif mode == 'UCB':
			self.f = self.UCB
		else:
			raise ValueError('Not recognised acquisition function')
	def ExpectedImprovement(self, xbest, mean, std):
		return norm.cdf((mean - xbest) / std)
	def ProbabilityImprovement(self, xbest, mean, std):
		z = (mean - xbest) / std
		return (mean - xbest) * norm.cdf(z) + std * norm.pdf(z)
	def UCB(self, xbest, mean, std, beta):
		return mean + beta * std
	def eval(self, xbest, mean, std):
		return self.f(xbest, mean, std, **self.params)
	
