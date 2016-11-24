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
	def ExpectedImprovement(self, tau, mean, std):
		return norm.cdf((mean - tau) / std)
	def ProbabilityImprovement(self, tau, mean, std):
		z = (mean - tau) / std
		return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)
	def UCB(self, tau, mean, std, beta):
		return mean + beta * std
	def eval(self, tau, mean, std):
		return self.f(tau, mean, std, **self.params)
	
